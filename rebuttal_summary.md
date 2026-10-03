# Meta-Review Point-by-Point: How We Answered

## 1. Focal-CKA is neither isolated in ablation nor functional in the code (GA discards it); visual-diversity contribution unverified (6Wre)

We address two separate claims here.

`run_ga.py` was leftover scaffolding never used for our reported results — the selection pipeline we actually ran uses a brute-force search that does include the Focal-CKA term. We are adding `run_brute_force.py` so this is directly reproducible.

To isolate Focal-CKA's contribution, we ran a full sweep of the text/vision weight ($w_1$ vs $w_2$) from 0 to 1, showing exactly which ensemble gets selected and how it scores under vision-only, text-only, and combined criteria on MMMU and A-OKVQA. Combining beats text-only on both datasets; vision-only alone is even the single best criterion on A-OKVQA (86.94% vs. 85.33%).

## 2. Empirical rigor lacking: single-seed results, unsupported significance claims, inconsistent headline metrics (6Wre, T7cB)

We reran the MLP/LED pipeline 10 times with different seeds and ran a one-sample t-test against the strongest base model (e.g., on MMMU: mean 53.50, t=8.663, p=0.000012).

We also found and fixed the metric inconsistency: our MMMU relative-gain number had been computed against the wrong denominator (fused accuracy instead of baseline). We recomputed every gain with one consistent formula, corrected +8.09% to +8.81%, and fixed the labeling of absolute vs. relative gain throughout the tables.

On LED appearing to underperform on MCQ tasks: this is expected behavior, not an inconsistency. LED targets open-ended tasks, and matching (not losing to) the base model on MCQ is fine given it leads on OCR-VQA instead.

## 3. Key baselines absent; included multi-agent baselines score below the weakest single model (xJZb, 6Wre)

We added six new baselines — self-consistency (Self-MoA, k=6, compute-matched), an LLM-Aggregator (gpt-4o-mini), MoA, Symbolic-MoE, MAgICoRe, and a retrained in-domain PairRanker — across MMMU, MMMU-Pro, A-OKVQA, and OCR-VQA. We also added majority/plurality voting plus an oracle upper bound.

The "baseline below weakest model" issue traced to our using an off-the-shelf, out-of-domain PairRanker checkpoint. Retraining it in-domain fixed it (22.36% to 53.29% on MMMU). V3Fusion still leads on almost every metric after these additions.

## 4. Epistemic-uncertainty decomposition over heterogeneous vocabularies lacks theoretical justification (6Wre, T7cB, 8H82)

We clarified that the decomposition never touches raw per-model token vocabularies — it operates over a shared answer space (fixed choices A/B/C/... for MCQ; an n=5-sample answer-frequency histogram for open-ended, the same estimator LLM-TOPLA uses), so cross-model vocabulary mismatch is a non-issue by construction. We grounded the identity in established theory: the Krogh-Vedelsby ambiguity decomposition, BALD, and Lakshminarayanan et al.'s framing of deep ensembles as posterior samples.

We also acknowledge this is still missing an actual calibration/reliability check — predicted uncertainty vs. empirical error — and we commit to adding one; we don't present the theoretical grounding as sufficient on its own.

## 5. Ablations fail to disentangle the trained fusion head from diversity-based pruning (T7cB)

We ran three experiments to isolate pruning's contribution from the head's:

1. Same compute, zero diversity (self-consistency, k=6) scores worse than a single model — so the trained/aggregating step alone doesn't explain our gains.
2. Holding the identical MLP fixed and only swapping which models get selected moves A-OKVQA accuracy by 1.6 points (85.33% to 86.94%) — isolating the selection criterion's effect.
3. We showed the diversity metric isn't just a stand-in for "pick the stronger model": CKA-diverse pairs have a smaller quality gap than average.

We're upfront that a fully controlled, fixed-head-vs-random-ensemble experiment across every dataset is still missing; we've flagged it for camera-ready.

---

## Summary of summary

We gave every meta-review point a substantive, data-backed response rather than a dismissal: we isolated Focal-CKA's contribution via a weight sweep and explained the code discrepancy as unused scaffolding; we added statistical rigor via 10-seed t-tests and a corrected, consistent gain formula; we added six missing baselines and fixed a mis-scoring one by retraining in-domain; we justified the uncertainty decomposition theoretically and showed the vocabulary concern is moot by construction; and we ran three targeted experiments separating the pruning criterion's effect from the trained fusion head's. Our weakest spots — where we flag remaining gaps rather than claiming full resolution — are the missing calibration curve for uncertainty estimates and the missing fully-controlled random-ensemble baseline, both of which we commit to for camera-ready.
