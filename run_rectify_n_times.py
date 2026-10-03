"""Runs `sft_weighted.py --rectify` several times with different seeds and summarizes the novel accuracy
before and after rectification (mean, std, standard error, 95% CI, and the paired before/after difference).

Any extra arguments are passed through to sft_weighted.py, e.g.
    python run_rectify_n_times.py --runs 10 -- --task_name mmmu --model_ids 1235 --rectify_policy vote
"""
import os
import sys
import argparse
import subprocess

import numpy as np
import torch
from scipy import stats


def summarize(name, values):
    n = len(values)
    mean, std = values.mean(), values.std(ddof=1)
    sem = std / np.sqrt(n)
    ci95 = stats.t.ppf(0.975, n - 1) * sem
    print(f"{name:<8} mean={mean:8.4f}  std={std:.4f}  sem={sem:.4f}  95%CI=±{ci95:.4f}  "
          f"min={values.min():.4f}  max={values.max():.4f}")
    return dict(mean=mean, std=std, sem=sem, ci95=ci95)


def main():
    parser = argparse.ArgumentParser(description="repeat sft_weighted.py --rectify and summarize accuracies")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--out_root", type=str, default="ensemble_rectify_runs",
                        help="each run is saved under results/<out_root>/seed_<seed>/<task_name>/<model_ids>")
    args, passthrough = parser.parse_known_args()
    passthrough = [a for a in passthrough if a != "--"]

    # sft_weighted.py's own defaults, only needed to locate the saved exp_result.pth
    sub_parser = argparse.ArgumentParser()
    sub_parser.add_argument("--task_name", default="okvqa")
    sub_parser.add_argument("--model_ids", default="123")
    sub_args, _ = sub_parser.parse_known_args(passthrough)

    before, after, seeds = [], [], []
    for i in range(args.runs):
        seed = args.start_seed + i
        out_root = f"{args.out_root}/seed_{seed}"
        print(f"=== Run {i + 1} / {args.runs} (seed {seed}) ===", flush=True)
        cmd = [sys.executable, "sft_weighted.py", "--rectify", "--seed", str(seed),
               "--out_root", out_root, *passthrough]
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"  WARNING: run with seed {seed} failed (exit code {ret.returncode}), skipping")
            continue

        result_path = os.path.join("results", out_root, sub_args.task_name, sub_args.model_ids, "exp_result.pth")
        rectify = torch.load(result_path, weights_only=False)["rectify"]
        before.append(rectify["before_acc"])
        after.append(rectify["after_acc"])
        seeds.append(seed)

    if len(before) < 2:
        print("Need at least 2 successful runs to compute uncertainty")
        return

    before, after = np.array(before), np.array(after)
    delta = after - before

    lines = [f"{'seed':>6} {'before':>10} {'after':>10} {'delta':>10}"]
    lines += [f"{s:>6} {b:>10.4f} {a:>10.4f} {d:>+10.4f}" for s, b, a, d in zip(seeds, before, after, delta)]
    print("\n" + "\n".join(lines))

    print(f"\n=== Summary ({len(before)} runs) ===")
    summarize("before", before)
    summarize("after", after)
    summarize("delta", delta)
    t_stat, p_value = stats.ttest_rel(after, before)
    print(f"paired t-test (after vs before): t={t_stat:.3f}, p={p_value:.4g}")

    out_path = os.path.join("results", args.out_root,
                            f"summary_{sub_args.task_name}_{sub_args.model_ids}.csv")
    np.savetxt(out_path, np.stack([seeds, before, after, delta], axis=1), delimiter=",",
               header="seed,before_acc,after_acc,delta", comments="", fmt=["%d", "%.4f", "%.4f", "%.4f"])
    print(f"Per-run results saved to {out_path}")


if __name__ == "__main__":
    main()
