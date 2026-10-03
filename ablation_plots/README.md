# Ablation figures

Figures that replace the ablation **tables** in `experiments.tex` and present the
new ablations run for the rebuttal.

```bash
python ablation_plots/make_ablation_plots.py                  # all 8 figures
python ablation_plots/make_ablation_plots.py --only 3 5       # just a couple
python ablation_plots/make_ablation_plots.py --outdir ../figures
```

Each figure is written to `figures/` as a **PDF** (include this one in LaTeX) and
a **PNG** (for quick viewing). Base font size is 16 pt, nothing below 12 pt.

To build the paper, copy the PDFs where the other figures live:

```bash
cp ablation_plots/figures/*.pdf <paper>/figures/
```

`experiments.tex` refers to them as `figures/ablation_*.pdf`.

## What each figure shows

| # | File | Replaces / adds | Source |
|---|------|-----------------|--------|
| 1 | `ablation_phase_gains` | **Table 5** — gain per phase | `experiments.tex` |
| 2 | `ablation_diversity_metrics` | **Table 6** — focal diversity vs. 5 error-correlation scores | `experiments.tex` |
| 3 | `ablation_weight_sweep` | new — $w_1$/$w_2$ sweep isolating Focal-CKA | review.md, R2-Q1 / R3-W1 |
| 4 | `ablation_selection_criterion` | new — CKA-only / Div-only / both, 5 seeds | `rebuttal_summary.md` |
| 5 | `ablation_seed_significance` | new — 40 seeds on 2 GPUs, $t$-tests | review.md, cross-hardware check |
| 6 | `ablation_cka_confound` | new — CKA is not a "strong + weak" proxy | review.md, R1-W2 |
| 7 | `ablation_ga_optimality` | new — GA vs. brute force at $N$ = 12 | review.md, R3-W3 / R4-W5 |
| 8 | `ablation_threshold_sensitivity` | new — $\tau$ across base pools | `results/threshold_sensitivity.csv` |

## Files

- `ablation_data.py` — every number, annotated with the section of `review.md`
  (or the results CSV) it came from. Edit numbers here, not in the plotting code.
- `make_ablation_plots.py` — one function per figure, shared style at the top.

Figure 8 reads `results/threshold_sensitivity.csv` directly; the other seven use
`ablation_data.py`, because the underlying scripts (`run_optimality_gap.py`,
`cka_confound_probe.py`, `cka_top5_quality_check.py`) print to stdout rather than
writing result files.

## Style

Colours are the validated colourblind-safe categorical palette; the blue ramp
(light → dark) is used wherever the three conditions have a natural order
(one component → the other → both), so the "both" condition always reads as the
darkest mark. Accuracy comparisons that need a zoomed axis are drawn as dots or
lines rather than bars, so no bar is ever truncated away from zero.

Abbreviations inside Figure 3: `IVL2-8B` = InternVL2-8B, `ds-tiny` /
`ds-small` = deepseek-vl2-tiny / -small, `Qwen2.5` = Qwen2.5-VL-7B-Instruct.
