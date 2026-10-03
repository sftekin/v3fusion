"""
Standalone extraction of the GMM-based epistemic-uncertainty rectification stage
originally prototyped in notebooks/entropy.ipynb, plus the base-model-pool
sensitivity analysis requested by a reviewer:

  "The method for determining the epistemic uncertainty threshold depends
   heavily on type of model being combined -- either Gaussian or 2 component
   GMM; using an arbitrary model as a base could create large variations in
   uncertainty thresholds."

We already have several MLP fusion heads trained on different base-model
subsets under results/ensemble/{dataset}/{model_ids}/exp_result.pth (used
for other ablations in the paper). This script reuses those subsets as
different "arbitrary" choices of base model pool, refits the adaptive
Gaussian-vs-GMM threshold on each one, and reports how much the threshold
tau (and the resulting rectified accuracy) moves around.

Only the original 6-model pool is used -- the 6 newly added base models are
intentionally excluded here.
"""
import os
import glob

import numpy as np
import pandas as pd
import torch
from scipy.stats import entropy, norm
from sklearn.mixture import GaussianMixture

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENSEMBLE_DIR = os.path.join(PARENT_DIR, "results", "ensemble")

# (dataset, answer-space size) -- MMMU/MMMU-Pro are 9-way MCQ, OK-VQA is 4-way.
DATASET_SPACE_SIZE = {
    "mmmu": 9,
    "mmmu_pro": 9,
    "okvqa": 4,
}


def load_ensemble_result(dataset_name, subset_dir=None):
    path = os.path.join(ENSEMBLE_DIR, dataset_name, subset_dir or "", "exp_result.pth")
    return torch.load(path, map_location="cpu", weights_only=False)


def list_subset_dirs(dataset_name):
    base = os.path.join(ENSEMBLE_DIR, dataset_name)
    subs = []
    for name in sorted(os.listdir(base)):
        full = os.path.join(base, name)
        if os.path.isdir(full) and os.path.isfile(os.path.join(full, "exp_result.pth")):
            subs.append(name)
    return subs


def compute_epistemic_uncertainty(logits, model_count, space_size):
    """Splits the stored (model_1 ... model_k, ensemble) logit block, computes
    per-model predictive entropy, ensemble predictive entropy, and the
    epistemic-uncertainty gap H(mean) - mean(H) used by V3Fusion-Rectify."""
    model_logits = logits[:, : model_count * space_size]
    ens_logits = logits[:, model_count * space_size:]
    assert ens_logits.shape[1] == space_size

    per_model_entropy = np.zeros(len(logits))
    mean_ens = np.zeros((len(logits), space_size))
    for model_prob in np.split(model_logits, model_count, axis=1):
        per_model_entropy += entropy(model_prob, base=2, axis=1)
        mean_ens += model_prob
    per_model_entropy /= model_count
    mean_ens /= model_count
    per_model_entropy[np.isnan(per_model_entropy)] = 0

    H_ens = entropy(mean_ens, base=2, axis=1)
    H_ens[np.isnan(H_ens)] = 0

    epistemic_uncertainty = H_ens - per_model_entropy
    return ens_logits, mean_ens, H_ens, per_model_entropy, epistemic_uncertainty


def adaptive_entropy_threshold(entropies, alpha=10.0):
    """Likelihood-ratio model-selection between a single Gaussian and a
    2-component GMM fit to the epistemic-uncertainty distribution; verbatim
    logic from notebooks/entropy.ipynb, factored into a reusable function."""
    entropies = np.array(entropies).reshape(-1, 1)

    mu, sigma = np.mean(entropies), np.std(entropies)
    sigma = max(sigma, 1e-8)
    logL1 = np.sum(norm.logpdf(entropies, mu, sigma))

    gmm = GaussianMixture(n_components=2, covariance_type="full",
                           reg_covar=1e-3, random_state=0)
    gmm.fit(entropies)
    logL2 = gmm.score(entropies) * len(entropies)

    llr = logL2 - logL1
    if llr > alpha:
        selected = "gmm"
        cluster_labels = gmm.predict(entropies)
        group0 = entropies[cluster_labels == 0]
        group1 = entropies[cluster_labels == 1]
        tau = min(group0.max(), group1.max())
    else:
        selected = "gaussian"
        tau = mu + 2 * sigma

    return float(tau), selected, float(llr), {"mu": float(mu), "sigma": float(sigma)}


def rectify_with_threshold(ens_logits, epistemic_uncertainty, tau):
    """Same rejection rule used in notebooks/entropy.ipynb cell 31: for
    high-uncertainty samples, zero out the ensemble's top pick so the vote
    falls through to the next-most-likely option."""
    ens_logit_copy = ens_logits.copy()
    reject_idx = epistemic_uncertainty > tau
    ens_logit_copy[reject_idx, ens_logits[reject_idx].argmax(1)] = 0
    return ens_logit_copy, reject_idx


def evaluate_subset(dataset_name, subset_dir, space_size, alpha=10.0):
    result = load_ensemble_result(dataset_name, subset_dir)
    model_names = result["model_names"]
    logits = result["logits"]
    labels = result["labels"]
    model_count = len(model_names)

    ens_logits, mean_ens, H_ens, per_model_entropy, epi_unc = compute_epistemic_uncertainty(
        logits, model_count, space_size)

    tau, selected, llr, gauss_params = adaptive_entropy_threshold(epi_unc, alpha=alpha)

    before_preds = ens_logits.argmax(1)
    before_acc = np.mean(labels == before_preds)

    rectified_logits, reject_idx = rectify_with_threshold(ens_logits, epi_unc, tau)
    after_acc = np.mean(labels == rectified_logits.argmax(1))

    return {
        "dataset": dataset_name,
        "subset": subset_dir or "(default)",
        "base_models": ", ".join(model_names),
        "pool_size": model_count,
        "selected_dist": selected,
        "log_lik_ratio": llr,
        "tau": tau,
        "epi_unc_mean": float(np.mean(epi_unc)),
        "epi_unc_std": float(np.std(epi_unc)),
        "reject_frac": float(np.mean(reject_idx)),
        "before_acc": before_acc,
        "after_acc": after_acc,
        "delta_acc": after_acc - before_acc,
    }


def run_sensitivity(dataset_name, alpha=10.0, include_default=True):
    space_size = DATASET_SPACE_SIZE[dataset_name]
    subset_dirs = list_subset_dirs(dataset_name)
    rows = []
    if include_default:
        rows.append(evaluate_subset(dataset_name, None, space_size, alpha))
    for sub in subset_dirs:
        rows.append(evaluate_subset(dataset_name, sub, space_size, alpha))
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.width", 160)
    all_rows = []
    for dataset_name in ["mmmu", "okvqa"]:
        print(f"\n=== {dataset_name} : base-model-pool sensitivity of the adaptive threshold ===")
        df = run_sensitivity(dataset_name)
        print(df[["subset", "base_models", "pool_size", "selected_dist",
                   "tau", "reject_frac", "before_acc", "after_acc", "delta_acc"]].to_string(index=False))
        print(f"\ntau range: [{df['tau'].min():.4f}, {df['tau'].max():.4f}] "
              f"(spread = {df['tau'].max() - df['tau'].min():.4f}, "
              f"std = {df['tau'].std():.4f}, mean = {df['tau'].mean():.4f})")
        print(f"distribution selected: {df['selected_dist'].value_counts().to_dict()}")
        print(f"delta_acc range: [{df['delta_acc'].min():+.4f}, {df['delta_acc'].max():+.4f}]")
        all_rows.append(df)

    out = pd.concat(all_rows, ignore_index=True)
    out_path = os.path.join(PARENT_DIR, "results", "threshold_sensitivity.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
