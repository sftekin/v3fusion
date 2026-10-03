"""
Reviewer concern: Focal-CKA measures pairwise dissimilarity between vision
encoders, not the absolute richness of either encoder's features. A low-CKA
pair could be genuinely complementary, or it could just be one strong encoder
paired with an impoverished one -- CKA alone can't tell those apart.

This script probes each encoder's own visual representations directly (a
linear classifier on the mean-pooled visual features already used for CKA,
predicting the dataset's answer index -- zero extra labeling effort, on-task)
to get an absolute "richness" score (member_quality) per encoder. It then
regresses ensemble fusion_gain on member_quality and focal_cka jointly: if
focal_cka still explains fusion_gain after controlling for member_quality,
the diversity signal isn't just a stand-in for encoder quality.
"""
import os
import sys
import argparse
import itertools

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from configs import HF_CACHE
os.environ['HF_HOME'] = HF_CACHE

from run_ga import model_mapper_dict, load_hist_data
from ens_metrics import calc_div_acc
from cka_utils import calc_cka_matrix, calc_focal_cka, load_pooled_embeddings
from data_generator.data_loader import DataCreator


def compute_kept_indices(task_name, ds_split):
    """
    obtain_visual_embeddings.py extracts one embedding per raw MMMU example,
    unfiltered. inference.py, which produces the labels in load_hist_data,
    skips multi-image and open-ended examples -- and those drops are
    scattered through the iteration order, not a clean prefix. Replicate
    that same filter here so the visual embeddings can be re-indexed to
    line up 1:1 with hist_data's samples. Returns None when no filtering
    is needed (e.g. okvqa/ocr, where every example survives that filter).
    """
    if "mmmu" not in task_name:
        return None
    ds_creator = DataCreator(task_name)
    kept = []
    idx = 0
    for ds in ds_creator.get(ds_split):
        for example in ds:
            images = [example[f"image_{i}"] for i in range(1, 8) if example[f"image_{i}"] is not None]
            if len(images) == 1 and example.get("question_type", "multiple-choice") != "open":
                kept.append(idx)
            idx += 1
    return kept


def train_linear_probes(model_names, pooled_embeddings, labels, test_size=0.3, seed=0):
    print("\nTraining linear probes (mean-pooled visual features -> answer index):")
    probe_acc = {}
    _, class_counts = np.unique(labels, return_counts=True)
    can_stratify = class_counts.min() >= 2
    for mn, X in zip(model_names, pooled_embeddings):
        X_np = X.to(torch.float32).cpu().numpy()
        Xtr, Xte, ytr, yte = train_test_split(
            X_np, labels, test_size=test_size, random_state=seed,
            stratify=labels if can_stratify else None)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xtr, ytr)
        acc = clf.score(Xte, yte)
        probe_acc[mn] = acc
        print(f"  {mn}: linear-probe accuracy = {acc:.4f}")
    return probe_acc


def run(args):
    model_names = [model_mapper_dict[int(i)] for i in args.model_ids]
    num_models = len(model_names)
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    infer_dir = os.path.join(parent_dir, "results", "inference")

    hist_data = load_hist_data(model_names, infer_dir, args.dataset_name, args.ds_split)
    model_own_acc = hist_data["error_arr"].mean(axis=0)

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
    for ens_size in range(2, num_models + 1):
        for comb in itertools.combinations(range(num_models), ens_size):
            solution = np.zeros(num_models, dtype=int)
            solution[list(comb)] = 1

            ensemble_acc = calc_div_acc(solution, hist_data, [0, 1, 0])[0]
            best_member_acc = max(model_own_acc[i] for i in comb)
            fusion_gain = ensemble_acc - best_member_acc

            comb_probe_acc = [probe_acc[model_names[i]] for i in comb]
            member_quality = np.mean(comb_probe_acc)
            quality_gap = max(comb_probe_acc) - min(comb_probe_acc)
            focal_cka = 1 - calc_focal_cka(comb, cka_matrix)  # dissimilarity: higher = more diverse

            rows.append({
                "models": ", ".join(model_names[i] for i in comb),
                "ensemble_size": ens_size,
                "ensemble_acc": ensemble_acc,
                "best_member_acc": best_member_acc,
                "fusion_gain": fusion_gain,
                "member_quality": member_quality,
                "quality_gap": quality_gap,
                "focal_cka": focal_cka,
            })

    df = pd.DataFrame(rows)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 160)
    print(f"\n{len(df)} ensemble combinations:\n")
    print(df.to_string(index=False))

    print("\nOLS: fusion_gain ~ member_quality + focal_cka")
    model = smf.ols("fusion_gain ~ member_quality + focal_cka", data=df).fit()
    print(model.summary())

    pairs = df[df["ensemble_size"] == 2]
    gap_cka_corr = pairs["quality_gap"].corr(pairs["focal_cka"])
    print(f"\nPairs only: corr(focal_cka, quality_gap) = {gap_cka_corr:.4f}")
    print("Low |corr| here means low-CKA (diverse) pairs are not simply strong+weak "
          "encoder pairings -- i.e. the diversity signal isn't confounded with a quality gap.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Probe-based check for whether Focal-CKA is confounded with encoder richness")
    parser.add_argument('--dataset_name', default="okvqa", choices=["mmmu", "mmmu_pro", "okvqa", "ocr"])
    parser.add_argument('--model_ids', default="012345", type=str)
    parser.add_argument("--ds_split", type=str,
                        default="validation", choices=["test", "validation", "train"])
    parser.add_argument("--probe_test_size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    run(arguments)
