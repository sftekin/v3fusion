import os
import time
import argparse
import itertools

import numpy as np
import pandas as pd

from run_ga import model_mapper_dict, load_hist_data
from ens_metrics import calc_div_acc
from cka_utils import calc_cka_matrix, calc_focal_cka, load_pooled_embeddings


def run(args):
    model_names = [model_mapper_dict[int(i)] for i in args.model_ids]
    num_models = len(model_names)
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    infer_dir = os.path.join(parent_dir, "results", "inference")

    hist_data = load_hist_data(model_names, infer_dir, args.dataset_name, args.ds_split)
    raw_weights = [1, 1, 0]

    print("Computing pairwise CKA matrix for the model pool...")
    pooled_embeddings = load_pooled_embeddings(model_names, args.dataset_name, args.ds_split, parent_dir)
    cka_matrix = calc_cka_matrix(pooled_embeddings)
    del pooled_embeddings

    rows = []
    start_time = time.time()
    for ens_size in range(2, num_models + 1):
        for comb in itertools.combinations(range(num_models), ens_size):
            solution = np.zeros(num_models, dtype=int)
            solution[list(comb)] = 1

            focal_div, acc_score = calc_div_acc(solution, hist_data, raw_weights)
            focal_cka = 1 - calc_focal_cka(comb, cka_matrix)

            score = (args.focal_div_weight * focal_div
                     + args.acc_weight * acc_score
                     + args.cka_weight * focal_cka)
            if args.size_penalty:
                score -= 0.1 * ens_size / num_models

            rows.append({
                "models": ", ".join(model_names[i] for i in comb),
                "ensemble_size": ens_size,
                "focal_div": focal_div,
                "accuracy": acc_score,
                "focal_cka": focal_cka,
                "score": score,
            })
    elapsed = time.time() - start_time

    results_df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 160)
    print(f"\nEvaluated {len(results_df)} ensemble combinations in {elapsed:.2f}s\n")
    print(results_df.to_string(index=False))

    print("\nTop 15 ensemble combinations:")
    for rank, row in results_df.head(15).iterrows():
        print(f"#{rank + 1}: [{row['models']}] | Focal Diversity = {row['focal_div']:.4f}, "
              f"Accuracy = {row['accuracy']:.4f}, Focal CKA = {row['focal_cka']:.4f}, "
              f"Score = {row['score']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='brute-force ensemble search (no GA)')
    parser.add_argument('--dataset_name', default="okvqa", choices=["mmmu", "mmmu_pro", "okvqa", "ocr"])
    parser.add_argument("--focal_div_weight", default=0.25, type=float)
    parser.add_argument("--cka_weight", default=0.5, type=float)
    parser.add_argument("--acc_weight", default=0.25, type=float)
    parser.add_argument("--size_penalty", default=0, type=int, choices=[0, 1])
    parser.add_argument('--model_ids', default="012345", type=str)
    parser.add_argument("--ds_split", type=str,
                        default="validation", choices=["test", "validation", "train"])
    arguments = parser.parse_args()

    run(arguments)
