"""Sweeps the pruning weights of focal diversity (w1) and focal CKA (w2 = 1 - w1) through the whole run.py pipeline,
with a fixed accuracy weight (--acc_weight, 0.5 by default as in ens_pruning/run_weight_sweep.py). Focal CKA needs the
visual features of MMMU or OKVQA validation, so MMMU-Pro is pruned on its training data, MMMU validation, with
--prune_on train. Each (weights, seed) is one run.py experiment under
results/run_experiments/weight_sweep[_acc<acc_weight>]/<task_name>. Runs already in that folder's accuracies.csv are
skipped, so a stopped sweep resumes where it left off. Then the novel accuracy before and after rectification is
plotted per weight setting, mean and std over the seeds, for every task in the folder. Extra arguments are passed on
to run.py, and a task folder only takes runs made with the same ones, e.g.
    python weight_sweep.py --seeds 0 1 2 3 4
    python weight_sweep.py --acc_weight 0.3 --plot_only
    python weight_sweep.py --tasks mmmu_pro --acc_weight 0.3 --seeds 0 1 2 --out_root <folder> \
        --prune_on train --select_top_k 5 --select_temperature 0.01
"""
import os
import sys
import json
import time
import argparse
import subprocess

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs import RESULT_DIR

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_TITLES = {"mmmu": "MMMU", "okvqa": "OKVQA", "mmmu_pro": "MMMU-Pro"}
SERIES_COLORS = {"mmmu": "#2a78d6", "okvqa": "#eb6834", "mmmu_pro": "#1baf7a"}  # as in ens_pruning/plot_weight_sweep.py
MODEL_SHORT_NAMES = ["LLaVA-7B", "LLaVA-13B", "Qwen2.5-VL", "InternVL2", "DeepSeek-VL2-Tiny", "DeepSeek-VL2-Small"]
SURFACE, INK, INK_SECONDARY, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4de", "#c3c2b7"
METRICS = {"before": "before rectification", "after": "after rectification"}


def load_runs(out_root, tasks, acc_weight):
    frames = []
    for task in tasks:
        csv_path = os.path.join(PROJECT_DIR, RESULT_DIR, out_root, task, "accuracies.csv")
        if os.path.isfile(csv_path):
            runs = pd.read_csv(csv_path, dtype={"pool_ids": str, "model_ids": str})
            frames.append(runs[runs.acc_weight == acc_weight].assign(task=task))
    runs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return runs if len(runs) else None


def check_run_args(sweep_root, tasks, run_args):
    """keeps runs made with different run.py options out of the same task folder, and records them for the plots"""
    path = os.path.join(sweep_root, "run_args.json")
    saved = {}
    if os.path.isfile(path):
        with open(path) as f:
            saved = json.load(f)
    for task in tasks:
        # task folders swept before this file was kept had no extra options
        swept = os.path.isfile(os.path.join(sweep_root, task, "accuracies.csv"))
        if saved.get(task, [] if swept else run_args) != run_args:
            sys.exit(f"[sweep] {os.path.join(sweep_root, task)} holds runs made with run.py options "
                     f"{saved.get(task, [])}, not {run_args}; pass another --out_root")
        saved[task] = run_args
    os.makedirs(sweep_root, exist_ok=True)
    with open(path, "w") as f:
        json.dump(saved, f, indent=1)


def run_sweep(args, passthrough):
    check_run_args(os.path.join(PROJECT_DIR, RESULT_DIR, args.out_root), args.tasks, passthrough)
    runs = load_runs(args.out_root, args.tasks, args.acc_weight)
    done = set() if runs is None else set(zip(runs.task, runs.focal_div_weight, runs.cka_weight, runs.seed))
    # seeds outermost, so a partial sweep already covers every weight setting
    jobs = [(seed, task, w1) for seed in args.seeds for task in args.tasks for w1 in args.w1_grid]
    todo = [(seed, task, w1) for seed, task, w1 in jobs if (task, w1, round(1 - w1, 1), seed) not in done]
    print(f"[sweep] {len(jobs) - len(todo)} of {len(jobs)} runs already done, {len(todo)} to go", flush=True)

    failed, start = [], time.time()
    for k, (seed, task, w1) in enumerate(todo, start=1):
        w2 = round(1 - w1, 1)
        exp_name = f"fd{w1:.1f}_cka{w2:.1f}_seed{seed}_{time.strftime('%Y%m%d-%H%M%S')}"
        cmd = [sys.executable, "run.py", "--task_name", task, "--seed", str(seed), "--focal_div_weight", str(w1),
               "--cka_weight", str(w2), "--acc_weight", str(args.acc_weight), "--out_root", args.out_root,
               "--exp_name", exp_name, *passthrough]
        ret = subprocess.run(cmd, cwd=PROJECT_DIR)
        if ret.returncode == 0:
            last = load_runs(args.out_root, [task], args.acc_weight).iloc[-1]
            status = (f"selected {last.model_ids}, before {last.before_rectify_acc:.2f}, "
                      f"after {last.after_rectify_acc:.2f}")
        else:
            failed.append(exp_name)
            status = f"FAILED with exit code {ret.returncode}"
        elapsed = time.time() - start
        print(f"[sweep] {k}/{len(todo)} {task} {exp_name}: {status} | {elapsed / 60:.1f} min elapsed, "
              f"ETA {elapsed / k * (len(todo) - k) / 60:.1f} min", flush=True)
    if failed:
        print(f"[sweep] {len(failed)} runs failed: {failed}")


def summarize(runs):
    return (runs.groupby(["task", "focal_div_weight", "cka_weight"])
            .agg(n=("seed", "size"),
                 before_mean=("before_rectify_acc", "mean"), before_std=("before_rectify_acc", "std"),
                 after_mean=("after_rectify_acc", "mean"), after_std=("after_rectify_acc", "std"),
                 selected=("model_ids", lambda ids: ", ".join(f"{m} x{c}" for m, c in ids.value_counts().items())),
                 # "mixed" when no ensemble got most seeds, e.g. with run.py --select_top_k
                 modal_ids=("model_ids", lambda ids: ids.mode().iloc[0]
                            if ids.value_counts().iloc[0] > len(ids) / 2 else "mixed"),
                 unanimous=("model_ids", lambda ids: ids.nunique() == 1))
            .reset_index())


def label_ensembles(ax, rows):
    """names the ensemble most seeds selected over each stretch of weights that selects the same one, with a * when
    some seeds selected another one"""
    rows = rows.reset_index(drop=True)
    texts, stretch_start = [], 0
    for i in range(1, len(rows) + 1):
        if i < len(rows) and rows.modal_ids[i] == rows.modal_ids[stretch_start]:
            continue
        lo, hi = rows.focal_div_weight[stretch_start], rows.focal_div_weight[i - 1]
        label = rows.modal_ids[stretch_start]
        if label != "mixed":
            label = "{" + ",".join(label) + "}" + ("" if rows.unanimous[stretch_start:i].all() else "*")
        texts.append(ax.text((lo + hi) / 2, 0.97, label, transform=ax.get_xaxis_transform(), ha="center", va="top",
                             fontsize=8.5, color=INK_SECONDARY))
        if i < len(rows):
            ax.axvline((hi + rows.focal_div_weight[i]) / 2, color=GRID, linewidth=1, zorder=1)
        stretch_start = i
    return texts


def stagger(texts, gap=4):
    """moves a label that would overlap its left neighbour down to a second row, once the layout is final"""
    renderer = texts[0].figure.canvas.get_renderer()
    right_edges = [float("-inf"), float("-inf")]
    for text in texts:
        box = text.get_window_extent(renderer)
        row = 0 if box.x0 > right_edges[0] + gap else 1
        text.set_y(0.97 - 0.07 * row)
        right_edges[row] = box.x1


def plot(summary, tasks, metric, out_path, ylims, acc_weight, run_args):
    name = METRICS[metric]
    fig, axes = plt.subplots(1, len(tasks), figsize=(max(5.2 * len(tasks), 6.5), 4.8), dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    labels = []
    for ax, task in zip(axes[0], tasks):
        rows = summary[summary.task == task].sort_values("focal_div_weight")
        ax.set_facecolor(SURFACE)
        ax.errorbar(rows.focal_div_weight, rows[f"{metric}_mean"], yerr=rows[f"{metric}_std"].fillna(0),
                    color=SERIES_COLORS[task], linewidth=2, marker="o", markersize=8, markeredgecolor=SURFACE,
                    markeredgewidth=2, elinewidth=1.2, capsize=3, zorder=3)
        labels.append(label_ensembles(ax, rows))
        seeds = f"{rows.n.min()}" if rows.n.min() == rows.n.max() else f"{rows.n.min()}-{rows.n.max()}"
        ax.set_title(f"{TASK_TITLES[task]} ({seeds} seed{'' if seeds == '1' else 's'} per setting)", fontsize=11,
                     color=INK, loc="left")
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks([round(0.1 * i, 1) for i in range(11)])
        ax.set_ylim(*ylims[task])
        ax.set_xlabel(r"$w_1$ (Focal-Diversity), $w_2 = 1 - w_1$ (Focal-CKA)", color=INK)
        ax.grid(True, axis="y", color=GRID, linewidth=1, zorder=0)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8.5)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(AXIS)
    axes[0][0].set_ylabel("Novel-set accuracy (%)", color=INK)
    fig.suptitle(f"Fused accuracy {name} across pruning weights\naccuracy weight {acc_weight:g}, "
                 f"mean ± std over seeds", fontsize=12, color=INK)
    unanimous = "" if summary.unanimous.all() else ", * when not by every seed"
    if (summary.modal_ids == "mixed").any():
        unanimous += ", mixed when none got most seeds"
    notes = [f"Top labels: ensemble the pruning selected (model ids{unanimous})",
             "  ".join(f"{i} {mn}" for i, mn in enumerate(MODEL_SHORT_NAMES))]
    options = [f"{TASK_TITLES[task]}: {' '.join(run_args[task])}" for task in tasks if run_args.get(task)]
    if options:
        notes.append("run.py options  ·  " + "  ·  ".join(options))
    fig.text(0.5, 0.005, "\n".join(notes), fontsize=8, color=INK_SECONDARY, ha="center", va="bottom",
             linespacing=1.6)
    fig.tight_layout(rect=(0, 0.035 * len(notes), 1, 1))
    for texts in labels:
        stagger(texts)
    for ext in ["png", "pdf"]:
        fig.savefig(f"{out_path}.{ext}", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path}.png")


def plot_sweep(args):
    out_dir = os.path.join(PROJECT_DIR, RESULT_DIR, args.out_root)
    # every task in the folder, so sweeping one task doesn't drop the others from the plots
    tasks = [task for task in TASK_TITLES if os.path.isfile(os.path.join(out_dir, task, "accuracies.csv"))]
    runs = load_runs(args.out_root, tasks, args.acc_weight)
    if runs is None:
        print("[sweep] no runs to plot yet")
        return
    summary = summarize(runs)
    pd.set_option("display.width", 160)
    print(summary.drop(columns=["modal_ids", "unanimous"]).round(2).to_string(index=False))
    run_args = {}
    if os.path.isfile(os.path.join(out_dir, "run_args.json")):
        with open(os.path.join(out_dir, "run_args.json")) as f:
            run_args = json.load(f)
    summary.drop(columns=["modal_ids", "unanimous"]).to_csv(os.path.join(out_dir, "weight_sweep_summary.csv"),
                                                            index=False)

    # the same y range for a task in both plots, so before and after compare directly
    tasks = [task for task in tasks if task in set(summary.task)]
    ylims = {}
    for task in tasks:
        rows = summary[summary.task == task].fillna({"before_std": 0, "after_std": 0})
        lo = min((rows[f"{m}_mean"] - rows[f"{m}_std"]).min() for m in METRICS)
        hi = max((rows[f"{m}_mean"] + rows[f"{m}_std"]).max() for m in METRICS)
        pad = max(0.5, (hi - lo) * 0.12)
        ylims[task] = (lo - pad, hi + 3 * pad)  # headroom for the ensemble labels
    for metric in METRICS:
        plot(summary, tasks, metric, os.path.join(out_dir, f"weight_sweep_{metric}_rectification"), ylims,
             args.acc_weight, run_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="focal diversity / focal CKA pruning weight sweep over run.py",
                                     allow_abbrev=False)  # so run.py's --seed isn't taken for --seeds
    parser.add_argument("--tasks", nargs="+", default=["mmmu", "okvqa"], choices=["mmmu", "okvqa", "mmmu_pro"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--w1_grid", nargs="+", type=float, default=[round(0.1 * i, 1) for i in range(11)],
                        help="focal diversity weights w1, the focal CKA weight is 1 - w1")
    parser.add_argument("--acc_weight", default=0.5, type=float,
                        help="plurality-vote accuracy weight of the pruning, the same for every weight setting")
    parser.add_argument("--out_root", default=None, type=str,
                        help="the runs are saved under results/<out_root>/<task_name>, the plots in results/<out_root>; "
                             "run_experiments/weight_sweep_acc<acc_weight> by default")
    parser.add_argument("--plot_only", action="store_true", help="only plot the runs saved so far")
    arguments, run_args = parser.parse_known_args()
    if arguments.out_root is None:
        # the sweep with run.py's default accuracy weight 0.5 keeps the plain folder name
        arguments.out_root = "run_experiments/weight_sweep" + (
            "" if arguments.acc_weight == 0.5 else f"_acc{arguments.acc_weight:g}")
    if not arguments.plot_only:
        run_sweep(arguments, run_args)
    plot_sweep(arguments)
