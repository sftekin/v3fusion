import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt

from run_weight_sweep import dedup_breakpoints

SERIES_COLORS = {
    "mmmu": "#2a78d6",   # categorical slot 1 (blue)
    "okvqa": "#eb6834",  # categorical slot 2 (orange)
    "mmmu_pro": "#1baf7a",
    "ocr": "#eda100",
}


def plot_panel(ax, dataset, breakpoints, color):
    breakpoints = breakpoints.sort_values("w1_lo").reset_index(drop=True)

    for _, row in breakpoints.iterrows():
        ax.hlines(row["fused_accuracy"], row["w1_lo"], row["w1_hi"],
                   color=color, linewidth=2, zorder=3)
    for i in range(len(breakpoints) - 1):
        cur, nxt = breakpoints.iloc[i], breakpoints.iloc[i + 1]
        ax.plot([cur["w1_hi"], nxt["w1_lo"]], [cur["fused_accuracy"], nxt["fused_accuracy"]],
                color=color, linewidth=2, zorder=3)

    for _, row in breakpoints.iterrows():
        mid = (row["w1_lo"] + row["w1_hi"]) / 2
        ax.plot(mid, row["fused_accuracy"], marker="o", markersize=7,
                markerfacecolor=color, markeredgecolor="#fcfcfb", markeredgewidth=1, zorder=4)
        ax.annotate(f"{row['fused_accuracy']:.3f}", (mid, row["fused_accuracy"]),
                    textcoords="offset points", xytext=(0, 9), fontsize=9,
                    color="#0b0b0b", ha="center")

    ax.axvline(0.0, color="#c3c2b7", linewidth=1, linestyle=":", zorder=1)
    ax.axvline(1.0, color="#c3c2b7", linewidth=1, linestyle=":", zorder=1)

    ax.set_title(dataset, fontsize=12, color="#0b0b0b", loc="left")
    ax.set_xlim(-0.05, 1.05)
    y_vals = breakpoints["fused_accuracy"]
    pad = max(0.02, (y_vals.max() - y_vals.min()) * 0.35)
    ax.set_ylim(y_vals.min() - pad, y_vals.max() + pad * 1.6)
    ax.grid(True, color="#e5e4de", linewidth=0.8, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#c3c2b7")
    ax.set_xlabel(r"$w_1$ (Focal-Diversity), $w_2 = 1 - w_1$ (Focal-CKA)")


def plot(csv_path, out_path):
    df = pd.read_csv(csv_path)
    breakpoints_df = dedup_breakpoints(df)
    datasets = list(df["dataset"].unique())

    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.5), dpi=150)
    if len(datasets) == 1:
        axes = [axes]
    fig.patch.set_facecolor("#fcfcfb")

    for ax, dataset in zip(axes, datasets):
        ax.set_facecolor("#fcfcfb")
        color = SERIES_COLORS.get(dataset, "#4a3aa7")
        plot_panel(ax, dataset, breakpoints_df[breakpoints_df["dataset"] == dataset], color)

    axes[0].set_ylabel("Fused (pruned ensemble) accuracy")
    fig.suptitle("Pruning + fusion accuracy vs. text/vision diversity weighting", fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.text(0.5, 0.005,
              "left edge = vision-only (Focal-CKA)   •   right edge = text-only (Focal-Diversity)",
              fontsize=8.5, color="#52514e", ha="center")
    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    return breakpoints_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser.add_argument("--csv", default=os.path.join(parent_dir, "results", "weight_sweep_results.csv"))
    parser.add_argument("--out", default=os.path.join(parent_dir, "results", "figures", "weight_sweep_fused_accuracy.png"))
    args = parser.parse_args()
    plot(args.csv, args.out)
