"""
GA vs. brute-force optimality-gap analysis for pruning at a larger pool size (N=12).

Merges the original 6-model MMMU pool (results/inference/mmmu/validation) with
6 newly added base models (results/inference_base_models/*), aligns them on the
subset of MMMU-validation questions common to both runs (the two runs used
different sample counts, 805 vs 900), then:
  1. runs an exhaustive brute-force search over all size>=2 subsets of the 12
     models (Focal-Diversity + accuracy objective, no CKA term since pooled
     visual embeddings aren't available for the new models), and
  2. runs the same GA used elsewhere in the repo (pygad, same hyperparameters
     as run_ga.py) multiple times,
comparing the GA's best score/selected-ensemble accuracy against the true
brute-force optimum to report an optimality gap rather than only search time.
"""
import os
import re
import time
import itertools

import numpy as np
import pandas as pd
import pygad

from run_ga import model_mapper_dict, load_hist_data as load_old_hist_data
from ens_metrics import calc_div_acc

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NEW_MODEL_DIRS = {
    "Qwen2-VL-72B-Instruct": 6,
    "Qwen3-VL-235B-A22B-Instruct": 7,
    "InternVL3_5-241B-A28B": 8,
    "granite-vision-4.1-4b": 9,
    "pixtral-12b-2409": 10,
    "gemma-4-31B-it": 11,
}
NEW_FINAL_CSV = "mmmu_validation_results_20260727_131712_final.csv"


def extract_letter_old(output):
    """Same extraction logic as run_ga.py's load_hist_data, for the direct-answer prompts."""
    def extract_letter(text):
        match = re.search(r"\((\w)\)", text)
        return match.group(1) if match else ""

    pred_txt = str(output)[:10].strip()
    if "\n" in pred_txt:
        pred_txt = pred_txt.split("\n")[1]
    if "(" in pred_txt or ")" in pred_txt:
        pred_txt = extract_letter(pred_txt)
    return pred_txt[:1].upper()


def extract_letter_new(text):
    """Extraction for rationale-then-answer prompts, with fallbacks for reasoning
    models that box the answer or say 'final answer' instead of 'Answer:'."""
    text = str(text)
    m = re.findall(r"Answer:\s*\(?([A-Za-z])\)?", text, flags=re.IGNORECASE)
    if m:
        return m[-1].upper()
    m = re.findall(r"\\boxed\{\\?text\{?\(?([A-Za-z])\)?", text)
    if m:
        return m[-1].upper()
    m = re.findall(r"\\boxed\{\(?([A-Za-z])\)?\}", text)
    if m:
        return m[-1].upper()
    m = re.findall(r"[Ff]inal [Aa]nswer[:\s]+is?\s*\(?([A-Za-z])\)?", text)
    if m:
        return m[-1].upper()
    return ""


def core_question(q, is_new):
    q = str(q)
    if is_new:
        idx = q.find("\nA.")
        if idx == -1:
            idx = q.find("\n")
        return q[:idx].strip() if idx != -1 else q.strip()
    return q.strip()


def build_merged_hist_data():
    old_names = [model_mapper_dict[i] for i in range(6)]
    infer_dir = os.path.join(PARENT_DIR, "results", "inference")

    # --- old 6 models: reuse repo's own text-extraction convention ---
    old_df0 = pd.read_csv(os.path.join(infer_dir, "mmmu", "validation", f"{old_names[0]}_output.csv"))
    old_core = old_df0["question"].map(lambda q: core_question(q, is_new=False))
    old_labels = old_df0["answer"].astype(str).values

    old_preds = {}
    for mn in old_names:
        df = pd.read_csv(os.path.join(infer_dir, "mmmu", "validation", f"{mn}_output.csv"))
        old_preds[mn] = df["generated_outputs"].map(extract_letter_old).values

    # --- new 6 models: shared question order across all six ---
    new_df0 = pd.read_csv(os.path.join(
        PARENT_DIR, "results", "inference_base_models", list(NEW_MODEL_DIRS)[0], NEW_FINAL_CSV))
    new_core = new_df0["question"].map(lambda q: core_question(q, is_new=True))
    new_labels = new_df0["ground_truth"].astype(str).values

    new_preds = {}
    for mn in NEW_MODEL_DIRS:
        df = pd.read_csv(os.path.join(PARENT_DIR, "results", "inference_base_models", mn, NEW_FINAL_CSV))
        new_preds[mn] = df["model_output"].map(extract_letter_new).values

    # --- align on questions common to both runs, dropping any duplicated stems ---
    old_first = ~old_core.duplicated(keep=False)
    new_first = ~new_core.duplicated(keep=False)
    old_key_to_idx = {k: i for i, k in zip(old_core.index[old_first], old_core[old_first]) }
    new_key_to_idx = {k: i for i, k in zip(new_core.index[new_first], new_core[new_first]) }
    common_keys = sorted(set(old_key_to_idx) & set(new_key_to_idx))

    n = len(common_keys)
    num_models = 6 + len(NEW_MODEL_DIRS)
    model_names_all = old_names + list(NEW_MODEL_DIRS)
    error_arr = np.zeros((n, num_models), dtype=int)
    pred_arr = np.zeros((n, num_models), dtype=int)
    label_arr = np.zeros(n, dtype=int)

    mismatches = 0
    for row, key in enumerate(common_keys):
        oi, ni = old_key_to_idx[key], new_key_to_idx[key]
        ol, nl = old_labels[oi], new_labels[ni]
        if ol != nl:
            mismatches += 1
        label_arr[row] = ord(ol) - ord("A")
        for m, mn in enumerate(old_names):
            p = old_preds[mn][oi]
            error_arr[row, m] = int(p == ol)
            pred_arr[row, m] = (ord(p) - ord("A")) if len(p) == 1 and p.isalpha() else 99
        for m, mn in enumerate(NEW_MODEL_DIRS):
            p = new_preds[mn][ni]
            error_arr[row, 6 + m] = int(p == nl)
            pred_arr[row, 6 + m] = (ord(p) - ord("A")) if len(p) == 1 and p.isalpha() else 99

    print(f"Common aligned questions: {n} (label mismatches between old/new ground truth: {mismatches})")
    hist_data = {"error_arr": error_arr, "pred_arr": pred_arr, "label_arr": label_arr}
    per_model_acc = error_arr.mean(axis=0)
    for mn, acc in zip(model_names_all, per_model_acc):
        print(f"  {mn:35s} acc={acc:.4f}")
    return hist_data, model_names_all


def fitness(solution, hist_data, weights=(0.5, 0.5, 0)):
    if sum(solution) < 2:
        return -99.0
    return float(sum(calc_div_acc(np.array(solution), hist_data, list(weights))))


def brute_force(hist_data, num_models, weights=(0.5, 0.5, 0)):
    best_score, best_comb = -np.inf, None
    rows = []
    start = time.time()
    for size in range(2, num_models + 1):
        for comb in itertools.combinations(range(num_models), size):
            sol = np.zeros(num_models, dtype=int)
            sol[list(comb)] = 1
            score = fitness(sol, hist_data, weights)
            rows.append((comb, score))
            if score > best_score:
                best_score, best_comb = score, comb
    elapsed = time.time() - start
    return best_score, best_comb, elapsed, rows


def run_ga_once(hist_data, num_models, weights, seed):
    def fitness_function(ga_instance, solution, solution_idx):
        return fitness(solution, hist_data, weights)

    ga_params = {
        "num_generations": 1000,
        "num_parents_mating": 50,
        "sol_per_pop": 100,
        "num_genes": num_models,
        "fitness_func": fitness_function,
        "gene_space": [0, 1],
        "parent_selection_type": "sss",
        "crossover_type": "two_points",
        "gene_type": int,
        "mutation_by_replacement": False,
        "mutation_probability": 0.,
        "stop_criteria": ["saturate_100"],
        "random_seed": seed,
    }
    ga_instance = pygad.GA(**ga_params)
    start = time.time()
    ga_instance.run()
    elapsed = time.time() - start
    solution, solution_fitness, _ = ga_instance.best_solution()
    return solution, solution_fitness, elapsed


def main():
    hist_data, model_names = build_merged_hist_data()
    num_models = len(model_names)
    weights = (0.5, 0.5, 0)

    print(f"\nBrute-force over {num_models} models ({2**num_models - num_models - 1} valid subsets)...")
    bf_score, bf_comb, bf_time, _ = brute_force(hist_data, num_models, weights)
    bf_names = [model_names[i] for i in bf_comb]
    bf_focal_div, bf_acc = calc_div_acc(
        np.array([1 if i in bf_comb else 0 for i in range(num_models)]), hist_data, [1, 1, 0])
    print(f"Brute-force optimum: score={bf_score:.4f} acc={bf_acc:.4f} time={bf_time:.3f}s")
    print(f"  ensemble: {bf_names}")

    n_runs = 10
    gaps_score, gaps_acc, ga_times, hit_optimum = [], [], [], 0
    for seed in range(n_runs):
        sol, sol_fit, ga_time = run_ga_once(hist_data, num_models, weights, seed)
        sol = sol.astype(int)
        _, ga_acc = calc_div_acc(sol, hist_data, [1, 1, 0])
        gap_score = bf_score - sol_fit
        gap_acc = bf_acc - ga_acc
        gaps_score.append(gap_score)
        gaps_acc.append(gap_acc)
        ga_times.append(ga_time)
        if gap_score < 1e-9:
            hit_optimum += 1
        sel_names = [model_names[i] for i in range(num_models) if sol[i]]
        print(f"  seed={seed}: GA score={sol_fit:.4f} acc={ga_acc:.4f} "
              f"gap_score={gap_score:.4f} gap_acc={gap_acc:.4f} time={ga_time:.2f}s ensemble={sel_names}")

    print(f"\nOver {n_runs} GA runs: hit brute-force optimum in {hit_optimum}/{n_runs} runs")
    print(f"  mean score gap = {np.mean(gaps_score):.4f} (std {np.std(gaps_score):.4f})")
    print(f"  mean acc gap   = {np.mean(gaps_acc):.4f} (std {np.std(gaps_acc):.4f})")
    print(f"  mean GA time   = {np.mean(ga_times):.2f}s vs brute-force time = {bf_time:.2f}s "
          f"({(1 - np.mean(ga_times)/bf_time)*100:.1f}% {'reduction' if np.mean(ga_times) < bf_time else 'increase'})")


if __name__ == "__main__":
    main()
