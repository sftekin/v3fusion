import os
import itertools
import argparse

import numpy as np
import pandas as pd

from run_ga import model_mapper_dict, load_hist_data
from ens_metrics import calc_div_acc
from cka_utils import calc_cka_matrix, calc_focal_cka, load_pooled_embeddings

ACC_WEIGHT = 0.5
W1_GRID = [round(x, 1) for x in np.arange(0.0, 1.01, 0.1)]


def enumerate_subsets(model_names, hist_data, cka_matrix):
    num_models = len(model_names)
    raw_weights = [1, 1, 0]
    rows = []
    for ens_size in range(2, num_models + 1):
        for comb in itertools.combinations(range(num_models), ens_size):
            solution = np.zeros(num_models, dtype=int)
            solution[list(comb)] = 1
            focal_div, acc_score = calc_div_acc(solution, hist_data, raw_weights)
            focal_cka = 1 - calc_focal_cka(comb, cka_matrix)
            rows.append({
                "models": [model_names[i] for i in comb],
                "ensemble_size": ens_size,
                "focal_div": focal_div,
                "acc": acc_score,
                "focal_cka": focal_cka,
            })
    return rows


def best_for_weights(subset_rows, w1, w2, acc_weight):
    best = None
    for row in subset_rows:
        score = w1 * row["focal_div"] + acc_weight * row["acc"] + w2 * row["focal_cka"]
        if best is None or score > best["score"]:
            best = {**row, "score": score}
    return best


def condition_name(w1):
    if w1 == 1.0:
        return "text-only (Focal-Diversity)"
    if w1 == 0.0:
        return "vision-only (Focal-CKA)"
    return "both"


def dedup_breakpoints(df):
    """Collapse consecutive w1 samples that select the same ensemble/accuracy
    into a single row spanning the w1/w2 range over which that choice holds."""
    out_rows = []
    for dataset, group in df.groupby("dataset", sort=False):
        group = group.sort_values("w1_focal_div").reset_index(drop=True)
        seg_start = 0
        for i in range(1, len(group) + 1):
            same_as_start = (
                i < len(group)
                and group.loc[i, "models"] == group.loc[seg_start, "models"]
                and np.isclose(group.loc[i, "fused_accuracy"], group.loc[seg_start, "fused_accuracy"])
            )
            if same_as_start:
                continue
            seg = group.loc[seg_start:i - 1]
            row = seg.iloc[0].to_dict()
            w1_lo, w1_hi = seg["w1_focal_div"].min(), seg["w1_focal_div"].max()
            w2_lo, w2_hi = seg["w2_focal_cka"].min(), seg["w2_focal_cka"].max()
            row["w1_lo"], row["w1_hi"] = w1_lo, w1_hi
            row["w2_lo"], row["w2_hi"] = w2_lo, w2_hi
            row["w1_range"] = f"{w1_lo:.1f}" if w1_lo == w1_hi else f"{w1_lo:.1f}–{w1_hi:.1f}"
            row["w2_range"] = f"{w2_lo:.1f}" if w2_lo == w2_hi else f"{w2_lo:.1f}–{w2_hi:.1f}"
            row["is_first"] = seg_start == 0
            row["is_last"] = i == len(group)
            out_rows.append(row)
            seg_start = i
    return pd.DataFrame(out_rows)


def run(dataset_name, ds_split, model_ids, parent_dir):
    model_names = [model_mapper_dict[int(i)] for i in model_ids]
    infer_dir = os.path.join(parent_dir, "results", "inference")

    hist_data = load_hist_data(model_names, infer_dir, dataset_name, ds_split)

    pooled_embeddings = load_pooled_embeddings(model_names, dataset_name, ds_split, parent_dir)
    cka_matrix = calc_cka_matrix(pooled_embeddings)
    del pooled_embeddings

    subset_rows = enumerate_subsets(model_names, hist_data, cka_matrix)

    results = []
    for w1 in W1_GRID:
        w2 = round(1 - w1, 1)
        best = best_for_weights(subset_rows, w1, w2, ACC_WEIGHT)
        results.append({
            "dataset": dataset_name,
            "condition": condition_name(w1),
            "w1_focal_div": w1,
            "w2_focal_cka": w2,
            "acc_weight": ACC_WEIGHT,
            "ensemble_size": best["ensemble_size"],
            "models": ", ".join(best["models"]),
            "focal_div": best["focal_div"],
            "focal_cka": best["focal_cka"],
            "fused_accuracy": best["acc"],
        })
    return pd.DataFrame(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep w1 (Focal-Diversity, text) / w2 (Focal-CKA, vision) pruning weights and report fused accuracy"
    )
    parser.add_argument("--datasets", nargs="+", default=["mmmu", "okvqa"],
                         choices=["mmmu", "mmmu_pro", "okvqa", "ocr"])
    parser.add_argument("--model_ids", default="012345", type=str)
    parser.add_argument("--ds_split", default="validation", type=str,
                         choices=["train", "validation", "test"])
    parser.add_argument("--out_csv", default=None, type=str)
    args = parser.parse_args()

    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 160)

    all_dfs = []
    for ds in args.datasets:
        print(f"\n=== {ds} ({args.ds_split}) ===")
        df = run(ds, args.ds_split, args.model_ids, parent_dir)
        print(df.drop(columns=["dataset"]).to_string(index=False))
        all_dfs.append(df)

    final_df = pd.concat(all_dfs, ignore_index=True)
    out_csv = args.out_csv or os.path.join(parent_dir, "results", "weight_sweep_results.csv")
    final_df.to_csv(out_csv, index=False)
    print(f"\nSaved sweep results to {out_csv}")
