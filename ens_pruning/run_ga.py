import pygad
import numpy as np
import pandas as pd
import os
import re

from diversity_stats import calc_generalized_div, calc_stat_matrices
from ens_methods import voting
from ens_metrics import calc_div_acc
from cka_utils import calc_cka_matrix, calc_focal_cka, load_pooled_embeddings
import time
import argparse



model_mapper_dict = {
    0: "llava-v1.6-vicuna-7b-hf",
    1: "llava-v1.6-vicuna-13b-hf",
    2: "Qwen2.5-VL-7B-Instruct",
    3: "InternVL2-8B",
    4: "deepseek-vl2-tiny",
    5: "deepseek-vl2-small"
}



def load_hist_data(model_names, infer_dir, task_name, ds_split):
    def extract_letter(text):
        match = re.search(r"\((\w)\)", text)
        return match.group(1) if match else ""

    error_list, pred_list = [], []
    for mn in model_names:
        data_path = os.path.join(infer_dir, task_name, ds_split, f"{mn}_output.csv")
        data_df = pd.read_csv(data_path)

        arr_path = os.path.join(infer_dir, task_name, ds_split, f"{mn}_prob.npy")
        prob_arr = np.load(arr_path)

        start_chr = 'A'
        choices = []
        for i in range(prob_arr.shape[1]):
            choices.append(start_chr)
            start_chr = chr(ord(start_chr) + 1)

        prob_pred = []
        for i in np.argmax(prob_arr, axis=1):
            prob_pred.append(choices[i])
        prob_pred = np.array(prob_pred, dtype=str)

        generated_outputs = data_df["generated_outputs"].values

        extracted_outputs = []
        for output in generated_outputs:
            pred_txt = str(output)[:10].strip()
            if "\n" in pred_txt:
                pred_txt = pred_txt.split("\n")[1]
            if "(" in pred_txt or ")" in pred_txt:
                pred_txt = extract_letter(pred_txt)
            extracted_outputs.append(pred_txt[:1].upper())
        extracted_outputs = np.array(extracted_outputs)

        labels = data_df["answer"].values.astype(str)
        if task_name == "mmmu_pro" and mn != "llava-v1.6-vicuna-13b-hf":
            extracted_outputs = np.delete(extracted_outputs, (1017), axis=0)
            prob_pred = np.delete(prob_pred, (1017), axis=0)
            labels = np.delete(labels, (1017), axis=0)
            prob_arr = np.delete(prob_arr, (1017), axis=0)

        errors = labels == extracted_outputs.astype(str)
        error_list.append(errors.astype(int))
        acc = np.mean(errors)
        
        print(prob_arr.shape)
        print(mn, acc, np.mean(labels.astype(str) == prob_pred))

        label_idx = []
        for i in range(len(labels)):
            label_idx.append(choices.index(labels[i]))
        pred_list.append(np.argmax(prob_arr, 1))

    hist_data = {
        "error_arr": np.array(error_list).T,
        "pred_arr": np.array(pred_list).T,
        "label_arr": np.array(label_idx).astype(int)
    }
    
    return hist_data


def replicate(in_arr, multp):
    if len(in_arr.shape) == 1:
        return in_arr
    return np.concatenate([in_arr for i in range(multp)], 1)


def run(args):

    model_names = [model_mapper_dict[int(i)] for i in args.model_ids]
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    infer_dir = os.path.join(parent_dir, "results" , "inference")
    hist_data = load_hist_data(model_names, infer_dir, args.dataset_name, args.ds_split)
    div_weights = [args.focal_div_weight, args.acc_weight, 0]
    size_penalty = args.size_penalty

    cka_matrix = None
    if args.cka_weight > 0:
        print("Computing pairwise CKA matrix for the model pool...")
        pooled_embeddings = load_pooled_embeddings(model_names, args.dataset_name, args.ds_split, parent_dir)
        cka_matrix = calc_cka_matrix(pooled_embeddings)
        del pooled_embeddings

    # errors_dict = {mn:hist_data["error_arr"][:, i] for i, mn in enumerate(model_names)}
    # stat_matrices = calc_stat_matrices(errors_dict)
    
    # hist_data = {k:np.repeat(v, 20, 0) for k, v in hist_data.items()}

    multiplier = 1
    if multiplier > 1:
        hist_data = {k:replicate(v, multiplier) for k, v in hist_data.items()}

    # the score of every subset the GA evaluates, so the report can rank unique ensembles
    fitness_cache = {}

    def fitness_function(ga_instance, solution, solution_idx):
        key = tuple(solution)
        if key in fitness_cache:
            return fitness_cache[key]
        if sum(solution) < 2:
            score = -99
        else:
            score = sum(calc_div_acc(solution, hist_data, div_weights))
            if args.cka_weight > 0:
                selected_ids = np.argwhere(solution).ravel()
                focal_cka = 1 - calc_focal_cka(selected_ids, cka_matrix)  # dissimilarity: higher = more diverse
                score += args.cka_weight * focal_cka
            if size_penalty:
                score -= 0.1 * sum(solution)/len(solution)
        fitness_cache[key] = score
        return score

    ga_params = {
        "num_generations": 1000,
        "num_parents_mating": 50,
        "sol_per_pop": 100,
        "num_genes": len(model_names) * multiplier,
        "fitness_func": fitness_function,
        "gene_space": [0, 1],
        "parent_selection_type": "sss",
        "crossover_type": "two_points",
        "gene_type": int,
        "mutation_by_replacement": False,
        "mutation_probability": 0.,
        # mutation_type="adaptive",
        # mutation_probability=[0.25, 0.01],
        # "stop_criteria": ["reach_0.475"],
        "stop_criteria": ["saturate_100"],
    }

    ga_instance = pygad.GA(**ga_params)
    print("Genetic algorithm has started")
    start_time = time.time()
    ga_instance.run()
    end_time = time.time()
    ga_instance.plot_fitness(ylabel="Score", title="", font_size=16)

    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    # the final population converges to copies of the best ensemble, so rank the unique ones the GA scored instead
    visited = sorted(((sol, fitness) for sol, fitness in fitness_cache.items() if sum(sol) >= 2),
                     key=lambda sol_fitness: sol_fitness[1], reverse=True)
    top_n = min(10, len(visited))
    report_weights = [1, 1, 0]
    print(f"\nTop {top_n} of the {len(visited)} unique ensemble sets the GA scored:")
    for rank, (sol, fitness) in enumerate(visited[:top_n], start=1):
        sol = np.array(sol)
        selected_models = [model_names[i] for i in range(len(model_names)) if sol[i]]
        focal_div, acc_score = calc_div_acc(sol, hist_data, report_weights)
        line = (f"#{rank}: {selected_models} | Focal Diversity = {focal_div:.4f}, "
                f"Accuracy = {acc_score:.4f}, Fitness = {fitness:.4f}")
        if args.cka_weight > 0:
            selected_ids = np.argwhere(sol).ravel()
            focal_cka = 1 - calc_focal_cka(selected_ids, cka_matrix)
            line += f", Focal CKA = {focal_cka:.4f}"
        print(line)

    print(f"Lasted {(end_time - start_time)}seconds")
    print(ga_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='focal diversity pruning')
    parser.add_argument('--dataset_name', default="mmmu", choices=["mmmu", "mmmu_pro", "okvqa", "ocr"])
    parser.add_argument("--focal_div_weight", default=0.5, type=float)
    parser.add_argument("--cka_weight", default=0, type=float)
    parser.add_argument("--acc_weight", default=0.5, type=float)
    parser.add_argument("--size_penalty", default=0, type=int, choices=[0, 1])
    parser.add_argument('--model_ids', default="012345", type=str)
    parser.add_argument("--ds_split", type=str,
                        default="validation", choices=["test", "validation", "train"])
    arguments = parser.parse_args()

    run(arguments)