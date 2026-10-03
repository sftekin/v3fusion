"""
Disentangles the trained fusion head from diversity-based pruning (meta-review
concern raised by T7cB): trains the *identical* MLP fusion procedure
(sft_weighted.py, same seed/train-test convention used throughout the paper)
on every possible 3-model subset of the original 6-model MMMU pool, then
checks where the Focal-Diversity(+CKA)-selected subsets rank among the full
population of same-size subsets.

If the selected subsets land at or near the top of that ranking, the
pruning criterion is doing real work beyond "any trained fusion head over
any subset" -- the fusion architecture/training recipe is literally
identical across every row, so only the choice of *which 3 models* differs.

Writes each run to results/ensemble_disentangle/rep{r}/mmmu/{model_ids}/ so
the existing results/ensemble/mmmu/* artifacts (used elsewhere in the
rebuttal) are never touched.
"""
import os
import itertools
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_ROOT_TMPL = "ensemble_disentangle/rep{rep}"
N_REPEATS = 3
TASK_NAME = "mmmu"

MODEL_NAMES = {
    0: "llava-v1.6-vicuna-7b-hf",
    1: "llava-v1.6-vicuna-13b-hf",
    2: "Qwen2.5-VL-7B-Instruct",
    3: "InternVL2-8B",
    4: "deepseek-vl2-tiny",
    5: "deepseek-vl2-small",
}

# Criterion-selected subsets from the Reviewer 2 Q1 sweep, for reference.
SELECTED = {
    "023": "Focal-Diversity + Focal-CKA (combined) winner",
    "235": "Focal-Diversity-only winner",
}


def run_one(model_ids, rep, seed=22, epochs=500):
    out_root = OUT_ROOT_TMPL.format(rep=rep)
    save_dir = os.path.join(PARENT_DIR, "results", out_root, TASK_NAME, model_ids)
    result_path = os.path.join(save_dir, "exp_result.pth")
    if os.path.isfile(result_path):
        return result_path
    cmd = [sys.executable, "sft_weighted.py",
           "--task_name", TASK_NAME,
           "--model_ids", model_ids,
           "--seed", str(seed),
           "--epochs", str(epochs),
           "--out_root", out_root]
    subprocess.run(cmd, cwd=PARENT_DIR, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return result_path


def main():
    combos = ["".join(str(i) for i in c) for c in itertools.combinations(range(6), 3)]
    print(f"{len(combos)} possible 3-model subsets: {combos}")

    rows = []
    for combo in combos:
        for rep in range(N_REPEATS):
            print(f"[{combo}] rep {rep}...", flush=True)
            result_path = run_one(combo, rep)
            d = torch.load(result_path, map_location="cpu", weights_only=False)
            rows.append({
                "model_ids": combo,
                "models": ", ".join(MODEL_NAMES[int(i)] for i in combo),
                "rep": rep,
                "val_acc": d["val_acc"],
                "test_acc": d["test_acc"],
            })
            pd.DataFrame(rows).to_csv(
                os.path.join(PARENT_DIR, "results", "disentangle_ablation_progress.csv"), index=False)

    df = pd.DataFrame(rows)
    summary = df.groupby(["model_ids", "models"])["test_acc"].agg(["mean", "std", "count"]).reset_index()
    summary = summary.sort_values("mean", ascending=False).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", None)
    print("\n=== All 20 3-model subsets, ranked by mean fused test accuracy (identical MLP recipe) ===")
    print(summary.to_string(index=False))

    print("\n=== Where do the criterion-selected subsets rank? ===")
    for model_ids, desc in SELECTED.items():
        row = summary[summary["model_ids"] == model_ids]
        if len(row):
            r = row.iloc[0]
            print(f"{model_ids} ({desc}): rank {int(r['rank'])}/{len(summary)}, "
                  f"mean acc = {r['mean']:.4f} (std {r['std']:.4f})")

    out_path = os.path.join(PARENT_DIR, "results", "disentangle_ablation_summary.csv")
    summary.to_csv(out_path, index=False)
    df.to_csv(os.path.join(PARENT_DIR, "results", "disentangle_ablation_raw.csv"), index=False)
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()
