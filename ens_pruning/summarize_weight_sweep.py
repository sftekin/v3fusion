import os
import argparse

import pandas as pd

from run_weight_sweep import dedup_breakpoints


def condition_label(row):
    # is_first always spans down to w1=0.0 (pure Focal-CKA); is_last always
    # spans up to w1=1.0 (pure Focal-Diversity) since the grid is sorted 0->1.
    if row["is_first"]:
        return "Vision-only (Focal-CKA)"
    if row["is_last"]:
        return "Text-only (Focal-Diversity)"
    return "Both (transition)"


def build_table(csv_path):
    df = pd.read_csv(csv_path)
    breakpoints_df = dedup_breakpoints(df)
    breakpoints_df = breakpoints_df.sort_values(["dataset", "w1_lo"])
    breakpoints_df["condition"] = breakpoints_df.apply(condition_label, axis=1)
    cols = ["dataset", "condition", "w1_range", "w2_range", "models",
            "ensemble_size", "focal_div", "focal_cka", "fused_accuracy"]
    return breakpoints_df[cols]


def to_markdown(table):
    header = ["Dataset", "Condition", "w1 (text)", "w2 (vision)", "Selected ensemble",
              "Size", "Focal-Div", "Focal-CKA", "Fused Acc."]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for _, r in table.iterrows():
        lines.append(
            f"| {r['dataset']} | {r['condition']} | {r['w1_range']} | {r['w2_range']} | "
            f"{r['models']} | {r['ensemble_size']} | {r['focal_div']:.3f} | "
            f"{r['focal_cka']:.3f} | {r['fused_accuracy']:.3f} |"
        )
    return "\n".join(lines)


def to_latex(table):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llllrrrr}",
        r"\toprule",
        r"Dataset & Condition & $w_1$ (text) & $w_2$ (vision) & Size & Focal-Div & Focal-CKA & Fused Acc. \\",
        r"\midrule",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"{r['dataset']} & {r['condition']} & {r['w1_range']} & {r['w2_range']} & "
            f"{r['ensemble_size']} & {r['focal_div']:.3f} & {r['focal_cka']:.3f} & "
            f"{r['fused_accuracy']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Pruning + fusion accuracy when sweeping the Focal-Diversity/Focal-CKA "
              r"mixing weight $w_1$/$w_2$ away from the 0.5/0.5 default. Only weight ranges that "
              r"select a distinct ensemble/accuracy are shown.}",
              r"\label{tab:weight_sweep}",
              r"\end{table}"]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser.add_argument("--csv", default=os.path.join(parent_dir, "results", "weight_sweep_results.csv"))
    parser.add_argument("--out_md", default=os.path.join(parent_dir, "results", "weight_sweep_table.md"))
    parser.add_argument("--out_tex", default=os.path.join(parent_dir, "results", "weight_sweep_table.tex"))
    args = parser.parse_args()

    table = build_table(args.csv)

    md = to_markdown(table)
    tex = to_latex(table)

    with open(args.out_md, "w") as f:
        f.write(md + "\n")
    with open(args.out_tex, "w") as f:
        f.write(tex + "\n")

    print(md)
    print(f"\nSaved markdown table to {args.out_md}")
    print(f"Saved LaTeX table to {args.out_tex}")
