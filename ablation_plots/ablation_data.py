"""
Numbers behind the V3Fusion rebuttal ablation figures.

Every entry carries the section of ``review.md`` (or the results CSV) it comes
from, so a figure can always be traced back to the experiment that produced it.
Keeping the data here rather than inline in the plotting code means the figures
and the LaTeX tables are driven by one source.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


# --------------------------------------------------------------------------
# Fig. 1 -- gains contributed by each phase (was Table 5)
# Source: experiments.tex, Table 5; absolute accuracy gain over the best base model.
# --------------------------------------------------------------------------
PHASE_GAINS = {
    "datasets": ["A-OKVQA", "MMMU", "MMMU-Pro"],
    "Pruning only": [1.07, 1.36, -1.79],
    "Fusion only": [-1.48, 3.40, 0.55],
    "V3Fusion (both)": [1.93, 4.54, 2.29],
}


# --------------------------------------------------------------------------
# Fig. 2 -- focal diversity against 5 pairwise error-correlation scores (was Table 6)
# Source: experiments.tex, Table 6. Accuracy of the ensemble each metric selects.
# --------------------------------------------------------------------------
DIVERSITY_METRICS = {
    "metrics": [
        "Fleiss",
        "Corr. Coef.",
        "Bin. Disag.",
        "Kappa",
        "Bin. Entropy",
        "Focal-Div.",
    ],
    "MMMU": [37.14, 37.14, 42.61, 37.14, 50.31, 54.50],
    "A-OKVQA": [82.79, 82.79, 79.04, 82.79, 76.77, 88.31],
}


# --------------------------------------------------------------------------
# Fig. 3 -- Focal-Diversity / Focal-CKA weight sweep
# Source: review.md, Reviewer 2 Q1 and Reviewer 3 W1.
# Subsets come from the brute-force sweep (results/weight_sweep_results.csv);
# the accuracy reported for each selected subset is its MLP-fused accuracy
# (sft_weighted.py, seed 22), not the plurality-vote proxy in that CSV.
# --------------------------------------------------------------------------
WEIGHT_SWEEP = {
    "MMMU": {
        "breaks": [(0.0, 0.6), (0.7, 0.7), (0.8, 1.0)],
        "acc": [50.40, 54.14, 53.95],
        "condition": ["Vision-only", "Both", "Text-only"],
        "ensemble": [
            "llava-13b\n+ IVL2-8B",
            "llava-7b\n+ Qwen2.5\n+ IVL2-8B",
            "Qwen2.5\n+ IVL2-8B\n+ ds-small",
        ],
        "size": [2, 3, 3],
    },
    "A-OKVQA": {
        "breaks": [(0.0, 0.4), (0.5, 0.8), (0.9, 1.0)],
        "acc": [86.94, 86.84, 85.33],
        "condition": ["Vision-only", "Both", "Text-only"],
        "ensemble": [
            "IVL2-8B\n+ ds-tiny",
            "IVL2-8B\n+ ds-small",
            "llava-13b\n+ Qwen2.5\n+ ds-small",
        ],
        "size": [2, 2, 3],
    },
}


# --------------------------------------------------------------------------
# Fig. 4 -- selection-criterion ablation, 5 seeds per cell
# Source: rebuttal_summary.md. Identical MLP fusion head throughout; only the
# pruning criterion that picks the ensemble changes.
# --------------------------------------------------------------------------
SELECTION_CRITERION_RUNS = {
    "MMMU": {
        "Focal-CKA only": [51.45, 50.72, 51.00, 50.18, 50.94],
        "Focal-Div. only": [54.22, 54.13, 54.79, 52.90, 53.95],
        # Weight sweep at focal_div_weight=0.9, cka_weight=0.1, fused accuracy
        # before rectification, seeds 0-4
        # (results/run_experiments/weight_sweep/mmmu/accuracies.csv).
        "Both": [55.0311, 56.7702, 57.0186, 54.9068, 55.4037],
    },
    "A-OKVQA": {
        "Focal-CKA only": [86.67, 86.32, 86.50, 86.41, 86.93],
        # Weight sweep at focal_div_weight=1.0, cka_weight=0.0, fused accuracy
        # before rectification, seeds 0-4
        # (results/run_experiments/weight_sweep/okvqa/accuracies.csv).
        "Focal-Div. only": [85.0655, 85.7642, 84.9782, 86.1135, 84.8035],
        "Both": [88.48, 88.57, 88.57, 88.57, 88.84],
    },
}


# --------------------------------------------------------------------------
# Fig. 9 -- MMMU-Pro Focal-Diversity / Focal-CKA weight sweep, before rectification
# Source: weight_sweep.py --tasks mmmu_pro --acc_weight 0.3 --seeds 0 1 2
#   --prune_on train --select_top_k 5 --select_temperature 0.01
# --------------------------------------------------------------------------
MMMU_PRO_SWEEP_CSV = os.path.join(
    RESULTS_DIR, "run_experiments", "weight_sweep_acc0.3_top5_T0.01", "mmmu_pro", "accuracies.csv")


# --------------------------------------------------------------------------
# Fig. 5 -- seed variance and significance, V3Fusion-MLP on MMMU
# Source: review.md, "Additional Evidence: Cross-Hardware Seed-Variance Check".
# 20 torch/numpy seeds per GPU architecture at the paper's fusion-head lr = 1e-3.
# --------------------------------------------------------------------------
SEED_RUNS = {
    "V100": [
        58.0691, 53.0763, 54.4309, 58.8682, 53.7974, 57.2830, 55.8407, 57.4909,
        54.6713, 55.6003, 54.6388, 55.5678, 57.4032, 51.6340, 55.6653, 55.3924,
        54.8792, 54.2230, 55.2397, 57.6111,
    ],
    "H100": [
        54.3984, 54.8792, 60.4860, 55.2397, 55.1520, 54.9116, 55.1520, 54.6388,
        54.7590, 54.3107, 58.9657, 57.7313, 55.5126, 58.5402, 54.3432, 55.3274,
        56.7698, 58.9982, 57.1628, 56.1460,
    ],
}

# Baselines the seed runs are tested against (same MMMU eval convention).
SEED_BASELINES = {
    "Strongest base member": 51.43,
    "PairRanker (in-domain)": 53.29,
}


# --------------------------------------------------------------------------
# Fig. 6 -- does Focal-CKA just rediscover "one strong + one weak"?
# Source: review.md, Reviewer 1 W2. Quality = held-out accuracy of a linear
# probe on each encoder's pooled visual features alone.
# --------------------------------------------------------------------------
CKA_TOP5_PAIRS = {
    "pairs": [
        "llava-13b +\nInternVL2-8B",
        "llava-7b +\nInternVL2-8B",
        "InternVL2-8B +\ndeepseek-tiny",
        "InternVL2-8B +\ndeepseek-small",
        "Qwen2.5-VL +\nInternVL2-8B",
    ],
    "focal_cka": [0.644, 0.615, 0.579, 0.552, 0.494],
    "quality_gap": [0.021, 0.012, 0.029, 0.017, 0.004],
    "top5_mean_gap": 0.0165,
    "all15_mean_gap": 0.0226,
}

# Per-encoder probe accuracy is reported in review.md only as the range 23-28%
# against a ~11% (1-of-9) random-guess floor, so the figure shows the band.
VISION_PROBE = {
    "low": 23.0,
    "high": 28.0,
    "chance": 100.0 / 9.0,
}


# --------------------------------------------------------------------------
# Fig. 7 -- GA optimality gap at a doubled pool (N = 12), MMMU
# Source: review.md, Reviewer 3 W3 / Reviewer 4 W5.
# --------------------------------------------------------------------------
POOL12_ACCURACY = {
    "llava-v1.6-vicuna-7b": 35.54,
    "llava-v1.6-vicuna-13b": 36.69,
    "Qwen2.5-VL-7B": 50.32,
    "InternVL2-8B": 51.08,
    "deepseek-vl2-tiny": 37.71,
    "deepseek-vl2-small": 31.72,
    "Qwen2-VL-72B": 61.15,
    "Qwen3-VL-235B-A22B": 75.92,
    "InternVL3.5-241B-A28B": 70.45,
    "granite-vision-4.1-4b": 47.52,
    "pixtral-12b-2409": 53.63,
    "gemma-4-31B-it": 78.34,
}

# The 6 models already in the paper's pool, for highlighting in the figure.
POOL12_ORIGINAL = [
    "llava-v1.6-vicuna-7b",
    "llava-v1.6-vicuna-13b",
    "Qwen2.5-VL-7B",
    "InternVL2-8B",
    "deepseek-vl2-tiny",
    "deepseek-vl2-small",
]

GA_VS_BRUTE_FORCE = {
    "n_subsets": 4083,
    "n_ga_runs": 10,
    "n_ga_matched": 10,
    "bf_score": 0.5587,
    "ga_score": 0.5587,
    "bf_accuracy": 76.43,
    "ga_accuracy": 76.43,
    "bf_time_s": 9.8,
    "ga_time_s": 0.95,
    "optimum": "llava-13b + Qwen3-VL-235B + gemma-4-31B",
    "runner_up_score_gap_pct": 0.13,
}


# --------------------------------------------------------------------------
# Fig. 8 -- threshold sensitivity across base pools
# Source: results/threshold_sensitivity.csv, produced by
# ens_pruning/uncertainty_rectify.py.
# --------------------------------------------------------------------------
THRESHOLD_CSV = os.path.join(RESULTS_DIR, "threshold_sensitivity.csv")


# --------------------------------------------------------------------------
# Table numbers kept here so the tables and figures cannot drift apart.
# Source: review.md, Reviewer 2 Q2 (corrected relative gain).
# --------------------------------------------------------------------------
RELATIVE_GAIN = {
    # metric: (best base = Qwen2.5-VL-7B, V3Fusion-Rectify, abs gain, rel %)
    "A-OKVQA": (87.24, 89.17, 1.93, 2.21),
    "MMMU": (51.55, 56.09, 4.54, 8.81),
    "MMMU-Pro": (46.98, 49.27, 2.29, 4.87),
    "OCR BLEU-1": (83.34, 85.71, 2.37, 2.84),
    "OCR EM": (72.00, 72.08, 0.08, 0.11),
    "OCR F1": (84.80, 86.57, 1.77, 2.09),
}
