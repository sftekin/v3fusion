"""Runs the whole pipeline for one task and stores it as an experiment under results/run_experiments/<task_name>:
  1. ensemble pruning: a genetic algorithm selects a subset of the model pool (ens_pruning/run_ga.py)
  2. fusion: an MLP fusion head is trained on the selected models and tested on the novel set (sft_weighted.py)
  3. rectification: epistemic-uncertainty rectification of the fused predictions on the novel set

Each experiment gets its own directory holding results.json (config, selected ensemble, accuracies), exp_result.pth
and best_model.tar, and appends a row of accuracies to results/run_experiments/<task_name>/accuracies.csv, e.g.
    python run.py --task_name mmmu --rectify_policy vote
"""
import os
import sys
import csv
import json
import time
import argparse

import numpy as np
import pygad
import torch
from torch.utils.data import DataLoader

from configs import RESULT_DIR
from data_generator.inference_loader import infer_dir, load_infer_prob_data, load_base_model_answers
from sft_weighted import model_mapper_dict, train_ensemble, rectify_predictions

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
# the ens_pruning modules import each other by their bare module names
sys.path.append(os.path.join(PROJECT_DIR, "ens_pruning"))
from run_ga import load_hist_data
from ens_metrics import calc_div_acc
from cka_utils import calc_cka_matrix, calc_focal_cka, load_pooled_embeddings

# genetic algorithm settings of ens_pruning/run_ga.py
GA_PARAMS = dict(num_generations=1000, num_parents_mating=50, sol_per_pop=100, gene_space=[0, 1],
                 parent_selection_type="sss", crossover_type="two_points", gene_type=int,
                 mutation_by_replacement=False, mutation_probability=0., stop_criteria=["saturate_100"])


def task_data(task_name, dataset_type):
    """(dataset, split) the fusion head is trained on and the novel (dataset, split) it is tested on, as in
    sft_weighted.py"""
    if task_name == "mmmu":
        return ("mmmu_pro", "test"), ("mmmu", "validation")
    if task_name == "mmmu_pro":
        return ("mmmu", "validation"), ("mmmu_pro", "test")
    return ("okvqa", dataset_type), ("okvqa", "validation")


def available_pool(pool_ids, sources):
    """Drops the pool models that have no inference outputs for one of the (dataset, split) sources, e.g. there are
    none for llava-v1.6-vicuna-7b-hf on okvqa/train, so no fusion head could be trained with it."""
    kept = ""
    for i in pool_ids:
        mn = model_mapper_dict[int(i)]
        missing = [f"{ds}/{split}" for ds, split in sources
                   if not all(os.path.isfile(os.path.join(infer_dir, ds, split, f"{mn}_{suffix}"))
                              for suffix in ["output.csv", "prob.npy"])]
        if missing:
            print(f"Dropping {mn} from the pool: no inference outputs for {', '.join(missing)}")
        else:
            kept += i
    if len(kept) < 2:
        raise ValueError(f"fewer than 2 models of the pool {pool_ids} have inference outputs for {sources}")
    return kept


def pool_cka_matrix(pool_ids, dataset, split):
    """Pairwise CKA of the pool's pooled visual embeddings. The raw embeddings take tens of GB to load, so the matrix
    is cached next to them and only recomputed when one of them is newer than the cache."""
    pool_names = [model_mapper_dict[int(i)] for i in pool_ids]
    embed_dir = os.path.join(PROJECT_DIR, RESULT_DIR, "visual_features", dataset, split)
    cache_path = os.path.join(embed_dir, f"cka_matrix_{pool_ids}.npy")
    embed_mtime = max(os.path.getmtime(os.path.join(embed_dir, f"{mn}_vis_embed.pt")) for mn in pool_names)
    if os.path.isfile(cache_path) and os.path.getmtime(cache_path) > embed_mtime:
        print(f"Using the CKA matrix cached in {cache_path}")
        return np.load(cache_path)
    print("Computing pairwise CKA matrix for the model pool...")
    cka_matrix = calc_cka_matrix(load_pooled_embeddings(pool_names, dataset, split, PROJECT_DIR))
    np.save(cache_path, cka_matrix)
    return cka_matrix


def prune_ensemble(args, pool_ids, dataset, split):
    """Phase 1, as in ens_pruning/run_ga.py: a genetic algorithm searches the pool for the subset with the best
    weighted sum of focal diversity, plurality-vote accuracy and (with --cka_weight) focal CKA dissimilarity. With
    --select_top_k k > 1 the ensemble is instead drawn among the k best, with probability proportional to fitness, or
    to exp((fitness - best) / T) with --select_temperature T."""
    pool_names = [model_mapper_dict[int(i)] for i in pool_ids]
    hist_data = load_hist_data(pool_names, infer_dir, dataset, split)
    div_weights = [args.focal_div_weight, args.acc_weight, 0]

    cka_matrix = pool_cka_matrix(pool_ids, dataset, split) if args.cka_weight > 0 else None

    def focal_cka(solution):
        return 1 - calc_focal_cka(np.argwhere(solution).ravel(), cka_matrix)  # dissimilarity: higher = more diverse

    # a score only depends on the subset, and the GA keeps revisiting the few subsets of a small pool
    fitness_cache = {}

    def fitness_function(ga_instance, solution, solution_idx):
        key = tuple(solution)
        if key not in fitness_cache:
            if sum(solution) < 2:
                score = -99
            else:
                score = sum(calc_div_acc(solution, hist_data, div_weights))
                if args.cka_weight > 0:
                    score += args.cka_weight * focal_cka(solution)
                if args.size_penalty:
                    score -= 0.1 * sum(solution) / len(solution)
            fitness_cache[key] = score
        return fitness_cache[key]

    ga_instance = pygad.GA(num_genes=len(pool_names), fitness_func=fitness_function, random_seed=args.seed,
                           **GA_PARAMS)
    print("Genetic algorithm has started")
    start_time = time.time()
    ga_instance.run()
    seconds = time.time() - start_time
    solution, fitness, _ = ga_instance.best_solution()

    def describe(solution, fitness):
        focal_div, vote_acc = calc_div_acc(np.asarray(solution), hist_data, [1, 1, 0])
        desc = dict(model_ids="".join(i for i, keep in zip(pool_ids, solution) if keep),
                    fitness=float(fitness), focal_div=float(focal_div), vote_acc=float(vote_acc) * 100)
        if args.cka_weight > 0:
            desc["focal_cka"] = float(focal_cka(solution))
        return desc

    visited = sorted(((s, f) for s, f in fitness_cache.items() if sum(s) >= 2), key=lambda sf: sf[1], reverse=True)
    top = [describe(s, f) for s, f in visited[:10]]
    print(f"\nTop {len(top)} of the {len(visited)} ensemble sets the GA visited:")
    for rank, d in enumerate(top, start=1):
        line = (f"#{rank}: {[model_mapper_dict[int(i)] for i in d['model_ids']]} | Focal Diversity = "
                f"{d['focal_div']:.4f}, Vote Acc = {d['vote_acc']:.2f}, Fitness = {d['fitness']:.4f}")
        if "focal_cka" in d:
            line += f", Focal CKA = {d['focal_cka']:.4f}"
        print(line)

    selected_rank = 1
    if args.select_top_k > 1:
        candidates = visited[:args.select_top_k]
        fitness_values = np.array([f for _, f in candidates])
        if args.select_temperature is not None:
            # softmax of the fitness: a small temperature favours the best, a large one draws near uniformly
            probs = np.exp((fitness_values - fitness_values.max()) / args.select_temperature)
        else:
            if (fitness_values <= 0).any():
                raise ValueError(f"fitness-proportional selection needs positive fitness, the top "
                                 f"{len(fitness_values)} have {fitness_values}")
            probs = fitness_values
        probs = probs / probs.sum()
        # a generator of its own, so the draw is reproducible per seed and leaves the global one (the data split) alone
        pick = int(np.random.default_rng(args.seed).choice(len(candidates), p=probs))
        solution, fitness = candidates[pick]
        selected_rank = pick + 1
        print(f"Drew #{selected_rank} of the top {len(candidates)}, selection probabilities "
              f"{', '.join(f'{p:.3f}' for p in probs)}")

    pruning = describe(solution, fitness)
    pruning.update(models=[model_mapper_dict[int(i)] for i in pruning["model_ids"]], data=f"{dataset}/{split}",
                   pool_ids=pool_ids, rank=selected_rank, generations=ga_instance.generations_completed,
                   seconds=seconds, top=top)
    print(f"Selected ensemble {pruning['model_ids']} {pruning['models']} (#{selected_rank}) after "
          f"{pruning['generations']} generations ({seconds:.1f}s)")
    return pruning


def train_fusion(args, model_names, data, novel_data, space_size, save_dir):
    """Phase 2, as in sft_weighted.py: an MLP fusion head is trained on the selected models' answer probabilities,
    keeping its best epoch on a 25% validation split, and tested on the novel set."""
    np.random.seed(args.seed)  # the same train/validation split sft_weighted.py makes with this seed
    data = data[np.random.permutation(len(data))]
    train_size = int(len(data) * 0.75)
    print(f"Train Size: {train_size}")
    train_loader = DataLoader(data[:train_size], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(data[train_size:], batch_size=args.batch_size, shuffle=True)
    novel_loader = DataLoader(novel_data, batch_size=args.batch_size, shuffle=False)
    return train_ensemble(model_names, train_loader, val_loader, novel_loader, n_epochs=args.epochs,
                          save_dir=save_dir, space_size=space_size)


def save_json(results, exp_dir):
    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


def append_csv_row(csv_path, row):
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})


def run(args):
    train_src, novel_src = task_data(args.task_name, args.dataset_type)
    prune_src = novel_src if args.prune_on == "novel" else train_src
    exp_name = args.exp_name or f"{time.strftime('%Y%m%d-%H%M%S')}_seed{args.seed}"
    exp_dir = os.path.join(PROJECT_DIR, RESULT_DIR, args.out_root, args.task_name, exp_name)
    os.makedirs(exp_dir)  # fails rather than overwrite an earlier experiment
    results = dict(exp_name=exp_name, config=vars(args))

    print(f"=== Phase 1: ensemble pruning on {'/'.join(prune_src)} ===")
    pool_ids = available_pool(args.pool_ids, [train_src, novel_src])
    pruning = prune_ensemble(args, pool_ids, *prune_src)
    results["pruning"] = pruning
    save_json(results, exp_dir)
    model_names = pruning["models"]

    data = load_infer_prob_data(model_names, *train_src)
    novel_data = load_infer_prob_data(model_names, *novel_src)
    space_size = (data.shape[1] - 1) // len(model_names)
    print(f"Space Size: {space_size}")
    router_answers = None
    if args.rectify_policy == "router":
        router_answers = load_base_model_answers(args.router_model, *novel_src, model_names[0])
        if len(router_answers) != len(novel_data):
            raise ValueError(f"{args.router_model} answers {len(router_answers)} samples, "
                             f"novel set has {len(novel_data)}")

    print(f"\n=== Phase 2: fusion training on {'/'.join(train_src)}, testing on {'/'.join(novel_src)} ===")
    fused = train_fusion(args, model_names, data, novel_data, space_size, exp_dir)

    print("\n=== Phase 3: rectification ===")
    rectify = rectify_predictions(fused["logits"], fused["labels"], len(model_names), space_size,
                                  alpha=args.rectify_alpha, policy=args.rectify_policy, router_answers=router_answers)
    fused.update(config=vars(args), pruning=pruning, rectify=rectify)
    torch.save(fused, os.path.join(exp_dir, "exp_result.pth"))

    accuracies = dict(prune_vote_acc=pruning["vote_acc"],
                      val_acc=float(fused["val_acc"]),
                      test_acc=float(fused["test_acc"]),
                      before_rectify_acc=float(rectify["before_acc"]),
                      after_rectify_acc=float(rectify["after_acc"]),
                      router_acc=None if rectify["router_acc"] is None else float(rectify["router_acc"]))
    results["accuracies"] = accuracies
    results["rectify"] = {k: v for k, v in rectify.items() if not isinstance(v, np.ndarray)}
    results["rectify"]["reject_rate"] = float(rectify["reject_idx"].mean() * 100)
    save_json(results, exp_dir)

    append_csv_row(os.path.join(os.path.dirname(exp_dir), "accuracies.csv"),
                   dict(exp_name=exp_name, seed=args.seed, prune_on=args.prune_on, pool_ids=pool_ids,
                        focal_div_weight=args.focal_div_weight, acc_weight=args.acc_weight,
                        cka_weight=args.cka_weight, model_ids=pruning["model_ids"],
                        rectify_policy=args.rectify_policy, **accuracies,
                        rectify_delta=accuracies["after_rectify_acc"] - accuracies["before_rectify_acc"],
                        reject_rate=results["rectify"]["reject_rate"],
                        detection_auroc=float(rectify["detection_auroc"]),
                        rejection_bacc=float(rectify["rejection"]["balanced_accuracy"])))

    print(f"\n=== Accuracies of {args.task_name} with {pruning['model_ids']}, saved to {exp_dir} ===")
    for name, value in accuracies.items():
        if value is not None:
            print(f"{name:<20} {value:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ensemble pruning, fusion training/testing and rectification')
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--task_name", type=str, default="okvqa",
                        choices=["okvqa", "mmmu", "mmmu_pro"])
    parser.add_argument("--dataset_type", type=str, default="train",
                        choices=["test", "validation", "train"],
                        help="okvqa split the fusion head is trained on")
    parser.add_argument('--out_root', default="run_experiments", type=str,
                        help="results/<out_root>/<task_name>/<exp_name> is used as the save dir")
    parser.add_argument('--exp_name', default=None, type=str,
                        help="name of the experiment directory, <date>-<time>_seed<seed> by default")
    # phase 1: ensemble pruning
    parser.add_argument('--pool_ids', default="012345", type=str,
                        help="model pool the genetic algorithm selects the ensemble from")
    parser.add_argument('--prune_on', default="novel", choices=["novel", "train"],
                        help="data the subsets are scored on: the novel set the fusion head is tested on (what "
                             "run_ga.py's default split amounts to, e.g. mmmu/validation for mmmu), or the data it "
                             "is trained on, which keeps the novel set out of the model selection")
    parser.add_argument("--focal_div_weight", default=0.5, type=float)
    parser.add_argument("--acc_weight", default=0.5, type=float)
    parser.add_argument("--cka_weight", default=0, type=float)
    parser.add_argument("--size_penalty", default=0, type=int, choices=[0, 1])
    parser.add_argument("--select_top_k", default=1, type=int,
                        help="draw the final ensemble among the k best ones the GA found, with probability "
                             "proportional to their fitness; 1 keeps the best")
    parser.add_argument("--select_temperature", default=None, type=float,
                        help="with --select_top_k, draw with probability proportional to exp((fitness - best) / T) "
                             "instead: a small T favours the best ensembles, a large T draws near uniformly")
    # phase 2: fusion training and testing
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    # phase 3: rectification
    parser.add_argument('--rectify_alpha', default=10.0, type=float,
                        help="log-likelihood-ratio margin for choosing the 2-component GMM over a single Gaussian")
    parser.add_argument('--rectify_policy', default="mean", choices=["mean", "vote", "second", "router"],
                        help="fallback for rejected samples: average of base-model probabilities, their majority vote, "
                             "the ensemble's second choice, or the answer of --router_model")
    parser.add_argument('--router_model', default="Qwen3-VL-235B-A22B-Instruct", type=str,
                        help="model under results/inference_base_models whose answers rejected samples are routed to")
    arguments = parser.parse_args()
    if arguments.select_temperature is not None and arguments.select_temperature <= 0:
        parser.error("--select_temperature must be positive (--select_top_k 1 always keeps the best)")
    run(arguments)
