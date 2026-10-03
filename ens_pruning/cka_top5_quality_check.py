"""
Direct version of the CKA-confound check (see cka_confound_probe.py for the
regression-based version): rank all model pairs by Focal CKA (dissimilarity --
higher means more diverse), take the top 5 most-diverse pairs, and show each
model's own linear-probe "quality" (richness of its visual representation)
side by side. If the top-5 pairs don't have a bigger quality gap than the
full pair pool on average, that's a direct demonstration that picking pairs
by low CKA isn't just picking out a strong encoder paired with a weak one.
"""
import os
import argparse
import itertools

import numpy as np
import pandas as pd

from run_ga import model_mapper_dict, load_hist_data
from cka_utils import calc_cka_matrix, calc_focal_cka, load_pooled_embeddings
from cka_confound_probe import compute_kept_indices, train_linear_probes


def run(args):
    model_names = [model_mapper_dict[int(i)] for i in args.model_ids]
    num_models = len(model_names)
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    infer_dir = os.path.join(parent_dir, "results", "inference")

    hist_data = load_hist_data(model_names, infer_dir, args.dataset_name, args.ds_split)

    print("Loading pooled visual embeddings and computing pairwise CKA matrix...")
    pooled_embeddings = load_pooled_embeddings(model_names, args.dataset_name, args.ds_split, parent_dir)

    kept_indices = compute_kept_indices(args.dataset_name, args.ds_split)
    if kept_indices is not None:
        print(f"Re-aligning visual embeddings to inference order "
              f"({len(kept_indices)} of {pooled_embeddings[0].shape[0]} raw samples kept)...")
        pooled_embeddings = [emb[kept_indices] for emb in pooled_embeddings]

    n_labels = len(hist_data["label_arr"])
    for mn, emb in zip(model_names, pooled_embeddings):
        assert emb.shape[0] == n_labels, (
            f"{mn}: {emb.shape[0]} visual embeddings vs {n_labels} inference labels -- "
            f"samples are not aligned, refusing to probe on mismatched data.")

    cka_matrix = calc_cka_matrix(pooled_embeddings)
    probe_acc = train_linear_probes(
        model_names, pooled_embeddings, hist_data["label_arr"],
        test_size=args.probe_test_size, seed=args.seed)
    del pooled_embeddings

    rows = []
    for i, j in itertools.combinations(range(num_models), 2):
        focal_cka = 1 - calc_focal_cka([i, j], cka_matrix)  # dissimilarity: higher = more diverse
        q_i, q_j = probe_acc[model_names[i]], probe_acc[model_names[j]]
        rows.append({
            "model_a": model_names[i],
            "model_b": model_names[j],
            "quality_a": q_i,
            "quality_b": q_j,
            "quality_gap": abs(q_i - q_j),
            "focal_cka": focal_cka,
        })

    df = pd.DataFrame(rows).sort_values("focal_cka", ascending=False).reset_index(drop=True)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 160)

    print(f"\nAll {len(df)} pairs, ranked by Focal CKA (dissimilarity, high = more diverse):\n")
    print(df.to_string(index=False))

    n_top = min(5, len(df))
    top = df.head(n_top)
    print(f"\nTop {n_top} most CKA-diverse pairs (what a pure-CKA selector would pick):")
    for rank, row in top.iterrows():
        print(f"#{rank + 1}: {row['model_a']} (quality={row['quality_a']:.3f}) + "
              f"{row['model_b']} (quality={row['quality_b']:.3f}) | "
              f"Focal CKA = {row['focal_cka']:.4f}, quality_gap = {row['quality_gap']:.4f}")

    top_gap = top["quality_gap"].mean()
    all_gap = df["quality_gap"].mean()
    print(f"\nMean quality_gap, top-{n_top} most diverse pairs : {top_gap:.4f}")
    print(f"Mean quality_gap, all {len(df)} pairs            : {all_gap:.4f}")
    if top_gap <= all_gap:
        print("-> The most CKA-diverse pairs are NOT more strong+weak skewed than the full pool "
              "average: low CKA here isn't just a proxy for a quality mismatch.")
    else:
        print("-> The most CKA-diverse pairs show a LARGER quality gap than the pool average: "
              "here the diversity signal does lean toward pairing mismatched-quality encoders.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Direct check: are the most CKA-diverse pairs just strong+weak pairings?")
    parser.add_argument('--dataset_name', default="mmmu", choices=["mmmu", "mmmu_pro", "okvqa", "ocr"])
    parser.add_argument('--model_ids', default="012345", type=str)
    parser.add_argument("--ds_split", type=str,
                        default="validation", choices=["test", "validation", "train"])
    parser.add_argument("--probe_test_size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    run(arguments)
