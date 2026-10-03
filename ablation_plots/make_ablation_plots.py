#!/usr/bin/env python
"""
Generate the V3Fusion ablation figures used in place of the ablation tables.

    python ablation_plots/make_ablation_plots.py            # all figures
    python ablation_plots/make_ablation_plots.py --only 3 5 # a subset
    python ablation_plots/make_ablation_plots.py --outdir /path/to/paper/figures

Each figure is written as both a PDF (for \\includegraphics) and a PNG (for
quick inspection). All text is set at >= 14 pt with a 16 pt base, so the figures
stay legible at the column widths used in the paper.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

import ablation_data as D

# --------------------------------------------------------------------------
# Style. Palette slots are the validated categorical set; the blue ramp is used
# wherever the three conditions have a natural order (component -> component ->
# both), and gray recedes for context marks.
# --------------------------------------------------------------------------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#4a3aa7"
RED = "#e34948"

RAMP = ["#86b6ef", "#3987e5", "#184f95"]  # light -> dark, ordinal

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8985"
GRID = "#dedcd6"
SURFACE = "#ffffff"

BASE_FS = 16

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": BASE_FS,
        "axes.titlesize": BASE_FS + 2,
        "axes.labelsize": BASE_FS,
        "xtick.labelsize": BASE_FS - 1,
        "ytick.labelsize": BASE_FS - 1,
        "legend.fontsize": BASE_FS - 1,
        "figure.titlesize": BASE_FS + 3,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.linewidth": 1.0,
        "text.color": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "grid.color": GRID,
        "grid.linewidth": 0.9,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def tidy(ax, axis="y", spines=("top", "right")):
    """Recessive chrome: hairline grid on one axis only, unused spines dropped."""
    for s in spines:
        ax.spines[s].set_visible(False)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(axis=axis, color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(OUTDIR, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  wrote {os.path.join(OUTDIR, name)}.{{pdf,png}}")


# ==========================================================================
# Fig. 1 -- gains contributed by each phase  (replaces Table 5)
# ==========================================================================
def fig_phase_gains():
    d = D.PHASE_GAINS
    datasets = d["datasets"]
    series = ["Pruning only", "Fusion only", "V3Fusion (both)"]
    x = np.arange(len(datasets))
    width = 0.26

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    tidy(ax)

    for i, (name, color) in enumerate(zip(series, RAMP)):
        vals = d[name]
        off = (i - 1) * width
        bars = ax.bar(
            x + off, vals, width * 0.9, label=name, color=color,
            edgecolor=SURFACE, linewidth=2.0, zorder=3,
        )
        for b, v in zip(bars, vals):
            va = "bottom" if v >= 0 else "top"
            pad = 0.14 if v >= 0 else -0.14
            ax.text(
                b.get_x() + b.get_width() / 2, v + pad, f"{v:+.2f}",
                ha="center", va=va, fontsize=BASE_FS - 2,
                color=INK if name == series[-1] else INK_2,
                fontweight="bold" if name == series[-1] else "normal", zorder=4,
            )

    ax.axhline(0, color=INK_2, linewidth=1.2, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Absolute accuracy gain (pts)")
    ax.set_ylim(-3.1, 5.3)
    ax.set_title("Each phase alone can hurt; together they never do", pad=14)
    ax.legend(loc="upper center", ncol=3, columnspacing=1.6, handlelength=1.4,
              bbox_to_anchor=(0.5, -0.11))

    ax.annotate(
        "pruning alone loses\n1.79 pts on MMMU-Pro",
        xy=(2 - width, -1.79), xytext=(1.03, -2.62),
        fontsize=BASE_FS - 3, color=INK_2, ha="center", va="center",
        linespacing=1.3,
        arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=1.2,
                        connectionstyle="arc3,rad=-0.2"),
    )
    save(fig, "ablation_phase_gains")


# ==========================================================================
# Fig. 2 -- focal diversity vs. 5 error-correlation scores  (replaces Table 6)
# ==========================================================================
def fig_diversity_metrics():
    d = D.DIVERSITY_METRICS
    metrics = d["metrics"]
    order = list(range(len(metrics)))[::-1]  # Focal-Div at the top

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
    for ax, ds, lim in zip(axes, ["MMMU", "A-OKVQA"], [(33, 58), (73, 92)]):
        tidy(ax, axis="x")
        vals = d[ds]
        y = np.arange(len(metrics))
        for j, idx in enumerate(order):
            best = metrics[idx].startswith("Focal-Div")
            color = BLUE if best else MUTED
            ax.hlines(j, lim[0], vals[idx], color=color,
                      linewidth=3.0 if best else 2.0, zorder=3)
            ax.plot(vals[idx], j, "o", markersize=13 if best else 10,
                    color=color, markeredgecolor=SURFACE, markeredgewidth=2,
                    zorder=4)
            ax.text(vals[idx] + (lim[1] - lim[0]) * 0.022, j, f"{vals[idx]:.2f}",
                    va="center", ha="left", fontsize=BASE_FS - 2,
                    color=INK if best else INK_2,
                    fontweight="bold" if best else "normal")

        ax.set_yticks(y)
        ax.set_yticklabels([metrics[i] for i in order])
        ax.set_xlim(*lim)
        ax.set_ylim(-0.7, len(metrics) - 0.3)
        ax.set_xlabel("Ensemble accuracy (%)")
        ax.set_title(ds, pad=10)

    fig.suptitle("Focal diversity selects a better ensemble than 5 pairwise "
                 "error-correlation scores", y=1.04)
    fig.tight_layout()
    save(fig, "ablation_diversity_metrics")


# ==========================================================================
# Fig. 3 -- Focal-Diversity / Focal-CKA weight sweep
# ==========================================================================
def fig_weight_sweep():
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.4))
    band_colors = ["#eaf2fd", "#f5f4f1", "#fdeee8"]

    for ax, ds in zip(axes, ["MMMU", "A-OKVQA"]):
        d = D.WEIGHT_SWEEP[ds]
        tidy(ax)
        accs = d["acc"]
        lo, hi = min(accs), max(accs)
        span = max(hi - lo, 1.0)
        ymin, ymax = lo - span * 0.90, hi + span * 0.50

        for (a, b), acc, cond, ens, col in zip(
            d["breaks"], accs, d["condition"], d["ensemble"], band_colors
        ):
            left, right = a - 0.05, b + 0.05
            mid = (left + right) / 2
            ax.axvspan(left, right, color=col, zorder=1)
            ax.hlines(acc, left, right, color=BLUE, linewidth=3.2, zorder=4)
            ax.plot([mid], [acc], "o", markersize=12, color=BLUE,
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
            ax.text(mid, acc + span * 0.07, f"{acc:.2f}", ha="center",
                    va="bottom", fontsize=BASE_FS - 1, fontweight="bold",
                    color=INK, zorder=6)
            ax.text(mid, ymax - span * 0.04, cond, ha="center", va="top",
                    fontsize=BASE_FS - 2, color=INK_2, zorder=6)
            ax.text(mid, ymin + span * 0.05, ens, ha="center", va="bottom",
                    fontsize=BASE_FS - 4, color=MUTED, linespacing=1.3,
                    zorder=6)

        best = int(np.argmax(accs))
        ax.plot([(d["breaks"][best][0] + d["breaks"][best][1]) / 2],
                [accs[best]], "o", markersize=19, markerfacecolor="none",
                markeredgecolor=BLUE, markeredgewidth=2.2, zorder=5)

        ax.axvline(0.5, color=INK_2, linewidth=1.3, zorder=3)
        ax.text(0.5, lo - span * 0.42, "paper default  $w_1\\!=\\!w_2\\!=\\!0.5$",
                rotation=90, ha="center", va="center", fontsize=BASE_FS - 4,
                color=INK_2, zorder=7,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE,
                          edgecolor="none", alpha=0.9))

        ax.set_xlim(-0.06, 1.06)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.set_xlabel("$w_1$  (text / Focal-Diversity weight)   $\\longrightarrow$")
        ax.set_ylabel("Fused accuracy (%, MLP head)")
        ax.set_title(ds, pad=10)

    fig.suptitle("Sweeping the vision/text pruning weight: the selected "
                 "ensemble changes, and Focal-CKA is never redundant", y=1.03)
    fig.tight_layout()
    save(fig, "ablation_weight_sweep")


# ==========================================================================
# Paper style shared by Figs. 4 and 9: boxed axes, dash-dot grid, framed legend,
# matching the verified / not-verified bar chart in the paper.
# ==========================================================================
PAPER_RED = "#b2182b"
PAPER_ORANGE = "#e08214"
PAPER_BLUE = "#2166ac"
PAPER_GRID = "#9e9e9e"
PAPER_GRAY = "#d6ccc0"  # warm grey, a baseline that should recede next to our method
PAPER_WARM = "#c8510a"  # burnt orange accent for our method


def paper_box(ax, grid_axis="y"):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(INK)
        s.set_linewidth(1.1)
    if grid_axis:
        ax.grid(axis=grid_axis, color=PAPER_GRID, linestyle="-.", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=4, width=1, color=INK, labelcolor=INK)


def paper_legend(ax, **kw):
    leg = ax.legend(frameon=True, fancybox=True, framealpha=1.0, edgecolor="#cccccc",
                    borderpad=0.5, **kw)
    leg.get_frame().set_linewidth(0.8)
    return leg


# ==========================================================================
# Fig. 4 -- selection-criterion ablation with seed variance
# ==========================================================================
def fig_selection_criterion():
    fig, ax_l = plt.subplots(figsize=(10.5, 5.6))
    ax_r = ax_l.twinx()
    paper_box(ax_l)
    paper_box(ax_r, grid_axis=None)
    conds = ["Focal-CKA only", "Focal-Div. only", "Both"]
    colors = [PAPER_RED, PAPER_ORANGE, PAPER_BLUE]
    datasets = ["MMMU", "A-OKVQA"]
    # MMMU reads off the left axis, A-OKVQA off the right one; both span six
    # tick steps so the right-axis ticks sit on the left axis' grid lines.
    axes = {"MMMU": (ax_l, (48, 60)), "A-OKVQA": (ax_r, (84.5, 90.5))}
    offsets = [-0.26, 0.0, 0.26]

    for g, ds in enumerate(datasets):
        ax, (ylo, yhi) = axes[ds]
        runs = D.SELECTION_CRITERION_RUNS[ds]
        for cond, color, dx in zip(conds, colors, offsets):
            a = np.array(runs[cond])
            m, sd = a.mean(), a.std(ddof=1)
            x = g + dx
            ax.errorbar(x, m, yerr=sd, fmt="s", markersize=11, color=color,
                        ecolor=color, elinewidth=2.2, capsize=7, capthick=2.2,
                        markeredgecolor=INK, markeredgewidth=0.8, zorder=5)
            ax.text(x, m + sd + (yhi - ylo) * 0.025, f"{m:.2f}\n$\\pm${sd:.2f}",
                    va="bottom", ha="center", fontsize=BASE_FS - 3, color=INK,
                    fontweight="bold" if cond == "Both" else "normal",
                    linespacing=1.15)
        ax.set_ylim(ylo, yhi)
        ax.set_yticks(np.linspace(ylo, yhi, 7))

    ax_l.axvline(0.5, color=INK, linewidth=1.1, zorder=1)
    ax_l.set_xlim(-0.5, len(datasets) - 0.5)
    ax_l.set_xticks(range(len(datasets)))
    ax_l.set_xticklabels(datasets, fontsize=BASE_FS + 1)
    ax_l.set_ylabel("MMMU acc. of trained ensemble (%)")
    ax_r.set_ylabel("A-OKVQA acc. of trained ensemble (%)")
    handles = [Line2D([], [], marker="s", linestyle="none", markersize=10,
                      color=c, markeredgecolor=INK, markeredgewidth=0.8, label=l)
               for c, l in zip(colors, conds)]
    paper_legend(ax_r, handles=handles, loc="upper left", handletextpad=0.3)
    ax_l.set_title("Ablation on Visual and Linguistic pruning choices", pad=10)
    fig.tight_layout()
    save(fig, "ablation_selection_criterion")


# ==========================================================================
# Fig. 5 -- seed variance and significance on MMMU
# ==========================================================================
def fig_seed_significance():
    runs = D.SEED_RUNS
    gpus = ["V100", "H100"]
    gpu_colors = [PAPER_RED, PAPER_BLUE]
    base_items = list(D.SEED_BASELINES.items())
    base_colors = [INK_2, PAPER_ORANGE]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14.0, 6.0))

    # ---- (a) accuracy of every seed, one box per GPU ----------------------
    paper_box(ax_a)
    data = [np.array(runs[g]) for g in gpus]
    bp = ax_a.boxplot(data, positions=range(len(gpus)), widths=0.45, patch_artist=True,
                      medianprops=dict(color=INK, linewidth=2.0), zorder=3)
    for i, color in enumerate(gpu_colors):
        bp["boxes"][i].set(facecolor=to_rgba(color, 0.3), edgecolor=color, linewidth=1.8)
        for line in bp["whiskers"][2 * i:2 * i + 2] + bp["caps"][2 * i:2 * i + 2]:
            line.set(color=color, linewidth=1.8)
        bp["fliers"][i].set(marker="o", markerfacecolor=color, markeredgecolor=INK,
                            markersize=7)
    for x, a in enumerate(data):
        ax_a.text(x, a.max() + 0.3, f"mean {a.mean():.2f}\n$\\pm$ {a.std(ddof=1):.2f} s.d.",
                  ha="center", va="bottom", fontsize=BASE_FS - 3, linespacing=1.2)
    for (name, val), color in zip(base_items, base_colors):
        ax_a.axhline(val, color=color, linestyle="--", linewidth=2.0, zorder=2,
                     label=f"{name}: {val:.2f}")
    ax_a.set_xticks(range(len(gpus)))
    ax_a.set_xticklabels(gpus)
    ax_a.set_xlim(-0.6, len(gpus) - 0.4)
    ax_a.set_ylim(50, 65)
    ax_a.set_xlabel("GPU architecture")
    ax_a.set_ylabel("MMMU accuracy (%)")
    ax_a.set_title(f"(a) V3Fusion-MLP accuracy, {len(data[0])} seeds per GPU", pad=10)
    paper_legend(ax_a, loc="upper left", title="Baselines", fontsize=BASE_FS - 3,
                 title_fontsize=BASE_FS - 3)

    # ---- (b) mean gain over each baseline, 95% CI and t-test p-value ------
    paper_box(ax_b)
    width = 0.36
    ci_handle = None
    for j, (gpu, color) in enumerate(zip(gpus, gpu_colors)):
        a = np.array(runs[gpu])
        n = a.size
        half = stats.t.ppf(0.975, n - 1) * a.std(ddof=1) / np.sqrt(n)
        for i, (name, val) in enumerate(base_items):
            x = i + (j - 0.5) * width
            gain = a.mean() - val
            p = stats.ttest_1samp(a, val).pvalue
            exp = int(np.floor(np.log10(p)))
            ax_b.bar(x, gain, width, color=color, edgecolor=INK, linewidth=0.8, zorder=3,
                     label=gpu if i == 0 else None)
            eb = ax_b.errorbar(x, gain, yerr=half, fmt="none", ecolor=INK, elinewidth=1.6,
                               capsize=6, capthick=1.6, zorder=4)
            ci_handle = ci_handle or eb
            ax_b.text(x, gain + half + 0.12,
                      f"+{gain:.2f}\n$p = {p / 10 ** exp:.1f}{{\\times}}10^{{{exp}}}$",
                      ha="center", va="bottom", fontsize=BASE_FS - 4, linespacing=1.3)
    ax_b.set_xticks(range(len(base_items)))
    ax_b.set_xticklabels([name for name, _ in base_items])
    ax_b.set_xlim(-0.6, len(base_items) - 0.4)
    ax_b.set_ylim(0, 7.5)
    ax_b.set_xlabel("Baseline")
    ax_b.set_ylabel("Mean accuracy gain (pts)")
    ax_b.set_title("(b) Gain over baselines (one-sample $t$-test)", pad=10)
    handles, labels = ax_b.get_legend_handles_labels()
    paper_legend(ax_b, handles=handles + [ci_handle], labels=labels + ["95% CI"],
                 loc="upper right", fontsize=BASE_FS - 3)

    fig.tight_layout(w_pad=2.5)
    save(fig, "ablation_seed_significance")


# ==========================================================================
# Fig. 6 -- is Focal-CKA just a proxy for "one strong + one weak"?
# ==========================================================================
def fig_cka_confound():
    d = D.CKA_TOP5_PAIRS
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.6),
                             gridspec_kw={"width_ratios": [1.45, 1.0]})

    # ---- (a) quality gap of the 5 most CKA-diverse pairs ------------------
    ax = axes[0]
    tidy(ax)
    x = np.arange(len(d["pairs"]))
    bars = ax.bar(x, d["quality_gap"], 0.58, color=BLUE, edgecolor=SURFACE,
                  linewidth=2, zorder=3)
    for b, v, c in zip(bars, d["quality_gap"], d["focal_cka"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.0008, f"{v:.3f}",
                ha="center", va="bottom", fontsize=BASE_FS - 3, color=INK)
        ax.text(b.get_x() + b.get_width() / 2, 0.0012, f"CKA\n{c:.3f}",
                ha="center", va="bottom", fontsize=BASE_FS - 5, color="#ffffff",
                linespacing=1.15, zorder=4)

    ax.axhline(d["all15_mean_gap"], color=ORANGE, linewidth=2.2, zorder=5)
    ax.text(len(x) - 0.42, d["all15_mean_gap"] + 0.0007,
            f"all 15 pairs: {d['all15_mean_gap']:.4f}", ha="right", va="bottom",
            fontsize=BASE_FS - 3, color=ORANGE, fontweight="bold")
    ax.axhline(d["top5_mean_gap"], color=AQUA, linewidth=2.2, zorder=5)
    ax.text(len(x) - 0.42, d["top5_mean_gap"] - 0.0008,
            f"top-5 CKA-diverse: {d['top5_mean_gap']:.4f}", ha="right",
            va="top", fontsize=BASE_FS - 3, color="#158a60", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(d["pairs"], fontsize=BASE_FS - 5, linespacing=1.2)
    ax.set_ylabel("Encoder quality gap")
    ax.set_ylim(0, 0.034)
    ax.set_title("(a)  CKA-diverse pairs are more evenly matched,\n"
                 "         not strong + weak", pad=10, loc="left",
                 fontsize=BASE_FS, linespacing=1.3)

    # ---- (b) every encoder carries real visual signal ---------------------
    ax = axes[1]
    tidy(ax)
    p = D.VISION_PROBE
    ax.bar([0], [p["chance"]], 0.5, color=MUTED, edgecolor=SURFACE,
           linewidth=2, zorder=3)
    ax.bar([1], [p["high"] - p["low"]], 0.5, bottom=p["low"], color=BLUE,
           edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.text(0, p["chance"] + 0.7, f"{p['chance']:.1f}%", ha="center",
            va="bottom", fontsize=BASE_FS - 2, color=INK_2)
    ax.text(1, p["high"] + 0.7, f"{p['low']:.0f}–{p['high']:.0f}%", ha="center",
            va="bottom", fontsize=BASE_FS - 2, color=INK, fontweight="bold")
    ax.annotate(
        "", xy=(1.42, p["high"]), xytext=(1.42, p["chance"]),
        arrowprops=dict(arrowstyle="<->", color=INK_2, linewidth=1.6),
    )
    ax.text(1.50, (p["chance"] + p["high"]) / 2, "2–2.5$\\times$\nchance",
            ha="left", va="center", fontsize=BASE_FS - 3, color=INK_2,
            linespacing=1.2)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Random guess\n(1 of 9)", "Linear probe on\neach encoder"],
                       fontsize=BASE_FS - 3, linespacing=1.3)
    ax.set_xlim(-0.6, 2.1)
    ax.set_ylim(0, 33)
    ax.set_ylabel("MMMU accuracy (%)")
    ax.set_title("(b)  Vision features alone, no question text", pad=10,
                 loc="left", fontsize=BASE_FS)

    fig.tight_layout()
    save(fig, "ablation_cka_confound")


# ==========================================================================
# Fig. 7 -- GA optimality gap at a doubled pool (N = 12)
# ==========================================================================
def fig_ga_optimality():
    g = D.GA_VS_BRUTE_FORCE
    times = [g["bf_time_s"], g["ga_time_s"]]

    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    paper_box(ax)
    bars = ax.bar([0, 1], times, 0.52, color=[PAPER_GRAY, PAPER_WARM], edgecolor=INK,
                  linewidth=0.8, zorder=3)
    for b, v in zip(bars, times):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.22, f"{v:.2f} s", ha="center",
                va="bottom", fontsize=BASE_FS - 1, fontweight="bold", color=INK)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Brute force\n({g['n_subsets']:,} subsets)", "GA\n(ours)"],
                       linespacing=1.3)
    ax.set_ylabel("Search time (s)")
    ax.set_ylim(0, 13.6)
    ax.set_xlim(-0.62, 1.72)
    ax.set_title(f"Same optimum, a tenth of the time ($N$ = {len(D.POOL12_ACCURACY)} models)",
                 pad=10)

    # arrow ends at the GA bar's top-left corner, clear of its value label
    ax.annotate(
        f"$-${100 * (1 - g['ga_time_s'] / g['bf_time_s']):.0f}%",
        xy=(1 - 0.28, g["ga_time_s"]), xytext=(0.5, 6.4), ha="center",
        fontsize=BASE_FS, color=PAPER_WARM, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=PAPER_WARM, linewidth=1.8,
                        connectionstyle="arc3,rad=0.2"),
    )
    ax.text(
        1.68, 12.9,
        f"identical ensemble in\n{g['n_ga_matched']}/{g['n_ga_runs']} GA runs\n"
        f"score gap  {g['bf_score'] - g['ga_score']:.4f}\n"
        f"accuracy gap  {g['bf_accuracy'] - g['ga_accuracy']:.2f} pts",
        ha="right", va="top", fontsize=BASE_FS - 3, color=INK, linespacing=1.45,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=SURFACE, edgecolor="#cccccc",
                  linewidth=0.8),
    )

    fig.tight_layout()
    save(fig, "ablation_ga_optimality")


# ==========================================================================
# Fig. 8 -- rectification threshold across base pools
# ==========================================================================
def fig_threshold_sensitivity():
    df = pd.read_csv(D.THRESHOLD_CSV)
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    tidy(ax)

    label = {"mmmu": "MMMU", "okvqa": "A-OKVQA"}
    rng = np.random.default_rng(3)
    for ds, color in zip(["mmmu", "okvqa"], [BLUE, ORANGE]):
        sub = df[df["dataset"] == ds]
        jitter = rng.uniform(-0.09, 0.09, size=len(sub))
        ax.plot(sub["pool_size"] + jitter, sub["tau"], "o", markersize=13,
                color=color, alpha=0.8, markeredgecolor=SURFACE,
                markeredgewidth=2, label=label[ds], zorder=4)
        lo, hi = sub["tau"].min(), sub["tau"].max()
        ax.text(5.35, (lo + hi) / 2 if ds == "mmmu" else hi,
                f"{label[ds]}\n$\\tau \\in$ [{lo:.3f}, {hi:.3f}]\n"
                f"{hi / lo:.1f}$\\times$ spread",
                ha="left", va="center", fontsize=BASE_FS - 3, color=color,
                linespacing=1.3)

    ax.annotate(
        "2-model pools sit low;\n3–5-model pools land\nin a tight band",
        xy=(2.1, 0.24), xytext=(2.75, 0.63), fontsize=BASE_FS - 3,
        color=INK_2, linespacing=1.35,
        arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=1.4,
                        connectionstyle="arc3,rad=-0.25"),
    )

    ax.set_xticks([2, 3, 4, 5])
    ax.set_xlim(1.55, 7.4)
    ax.set_ylim(0, 1.45)
    ax.set_xlabel("Base-pool size (number of VLMs)")
    ax.set_ylabel("Selected threshold  $\\tau$")
    ax.set_title("Adaptive threshold tracks pool size, not pool membership\n"
                 "(2-component GMM chosen in 16/16 pools)", pad=14,
                 fontsize=BASE_FS + 1, linespacing=1.3)
    ax.legend(loc="upper left", handletextpad=0.4)
    save(fig, "ablation_threshold_sensitivity")


# ==========================================================================
# Fig. 9 -- MMMU-Pro pruning-weight sweep, before rectification
# ==========================================================================
def fig_weight_sweep_mmmu_pro():
    runs = pd.read_csv(D.MMMU_PRO_SWEEP_CSV, dtype={"model_ids": str})
    rows = []
    for w, g in runs.groupby("focal_div_weight"):
        counts = g.model_ids.value_counts()
        # the ensemble most seeds selected, "mixed" when none got most of them
        modal = counts.index[0] if counts.iloc[0] > len(g) / 2 else "mixed"
        rows.append((w, g.before_rectify_acc.mean(), g.before_rectify_acc.std(),
                     modal, g.model_ids.nunique() == 1, len(g)))
    w, mean, sd, modal, unanimous, n = map(np.array, zip(*rows))

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    paper_box(ax)
    ax.errorbar(w, mean, yerr=sd, color=PAPER_BLUE, linewidth=2.2, marker="o",
                markersize=9, markeredgecolor=INK, markeredgewidth=0.8,
                elinewidth=1.6, capsize=5, capthick=1.6, zorder=3,
                label="Before rectification")

    # name the selected ensemble over each stretch of weights that picks the same one
    start = 0
    for i in range(1, len(w) + 1):
        if i < len(w) and modal[i] == modal[start]:
            continue
        label = modal[start]
        if label != "mixed":
            label = "{" + ",".join(label) + "}" + ("" if unanimous[start:i].all() else "*")
        ax.text((w[start] + w[i - 1]) / 2, 0.955, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=BASE_FS - 4, color=INK_2)
        if i < len(w):
            ax.axvline((w[i - 1] + w[i]) / 2, color=PAPER_GRID, linestyle="-.",
                       linewidth=0.8, zorder=1)
        start = i

    ax.set_xlim(-0.05, 1.05)
    # both weights under each tick, Focal-Div. on top and Focal-CKA = 1 - it below
    ticks = [round(0.1 * i, 1) for i in range(11)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.1f}\n{1 - t:.1f}" for t in ticks], linespacing=1.5)
    # row names laid out like the tick labels (same font, line spacing and
    # offset of tick length + pad below the axis) so the rows line up, and
    # anchored to the first tick so they sit just left of its label
    tick = ax.xaxis.get_major_ticks()[0]
    first = ax.get_xticklabels()[0]
    half_width = (first.get_window_extent(fig.canvas.get_renderer()).width / 2
                  * 72 / fig.dpi)
    ax.annotate("Focal-Div.\nFocal-CKA", xy=(ticks[0], 0), xycoords=ax.get_xaxis_transform(),
                xytext=(-(half_width + 8), -(tick.get_tick_padding() + tick.get_pad())),
                textcoords="offset points", ha="right", va="top", linespacing=1.5,
                fontsize=plt.rcParams["xtick.labelsize"], color=INK)
    ax.set_ylim(40, 54)
    ax.set_xlabel("Focal-Div. weight $\\uparrow$     Focal-CKA weight $\\downarrow$",
                  labelpad=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Weight sweep analysis in MMMU-Pro", pad=10)
    paper_legend(ax, loc="upper left", bbox_to_anchor=(0.0, 0.88))
    fig.tight_layout()
    save(fig, "ablation_weight_sweep_mmmu_pro")


FIGURES = {
    1: fig_phase_gains,
    2: fig_diversity_metrics,
    3: fig_weight_sweep,
    4: fig_selection_criterion,
    5: fig_seed_significance,
    6: fig_cka_confound,
    7: fig_ga_optimality,
    8: fig_threshold_sensitivity,
    9: fig_weight_sweep_mmmu_pro,
}


def main():
    global OUTDIR

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", type=int, nargs="+", choices=sorted(FIGURES),
                    help="render only these figure numbers")
    ap.add_argument("--outdir", default=OUTDIR, help="where to write the files")
    args = ap.parse_args()

    OUTDIR = args.outdir

    for num in args.only or sorted(FIGURES):
        print(f"[{num}] {FIGURES[num].__name__}")
        FIGURES[num]()


if __name__ == "__main__":
    main()
