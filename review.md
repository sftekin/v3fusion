# Reviewer 1, xJZb
We first respond to W2 and then respond to W3+W4, followed by response to W1. 

**W2. Impact of CKA-preferred pairs:** We respond to W2 by addressing three questions raised:

*First, we will show that CKA-selected pairs are not simply pairing a strong encoder with a weak one.*

We trained a simple linear probe on each encoder's own pooled visual features (the same features used for CKA) to predict the answer directly, without using the text of question (query) and without other model's help. The probe's held-out accuracy is the so-called encoder's **"quality."** 

For any pair of encoders, their "quality gap" is the difference between their two quality scores. If low CKA just meant "one strong, one weak," the most CKA-diverse pairs should show unusually large quality gaps. 

In V3Fusion, CKA-selected pairs are not simply pairing a strong encoder with a weak one. We below show that the top 5 most CKA-diverse pairs on MMMU:

| Rank | Pair | Focal CKA (higher = more diverse) | Quality gap |
|---|---|---|---|
| 1 | llava-13b + InternVL2-8B | 0.644 | 0.021 |
| 2 | llava-7b + InternVL2-8B | 0.615 | 0.012 |
| 3 | InternVL2-8B + deepseek-tiny | 0.579 | 0.029 |
| 4 | InternVL2-8B + deepseek-small | 0.552 | 0.017 |
| 5 | Qwen2.5-VL + InternVL2-8B | 0.494 | 0.004 |
| |

Average quality gap, top-5 pairs: **0.0165**. Average across all 15 pairs: **0.0226**. The most CKA-diverse pairs are slightly *more* evenly matched than the average, the opposite of what the "strong+weak" pair predicts.

*Next, we show that the CKA selected encoders do carry rich, informative visual representations in the first place.*

CKA only compares encoders to each other, so it is not used to certify the absolute quality of either encoder's features with respect to the query. Using the same probe per-encoder to predict the MMMU answer from vision features alone: random guessing is ~11% (9 choices), and every encoder scored 23–28%,  roughly 2–2.5x chance. This presents some direct evidence that each encoder retains real, task-relevant visual signal, not noise that merely looks "different" under CKA.

*Finally, we show that the pattern of how CKA-selected encoders play out in the trained ensembles may differ by dataset.*

| Selection criterion | MMMU acc. | OKVQA acc. |
|---|---|---|
| Focal-CKA only | 50.40 | 86.94 |
| Focal-Diversity only | 53.95 | 85.33 |
| Both combined | 54.14 | 86.32 |
||

On MMMU, Focal-CKA alone (50.40) is weaker than Focal-Diversity alone (53.95), and combining the two we gain slightly higher accuracy (54.14), and CKA does add a small gain on top of focal-diversity rather than replacing it. 

On OKVQA, the roles are reversed: Focal-CKA alone (86.94) is the strongest single criterion, beating Focal-Diversity alone (85.33) by 1.6 points, with the combined criterion (86.32) landing in between. 

Across both datasets, combining the two criteria (Focal-CKA and Focal-diversity) matches or outperforms Focal-Diversity alone; showing that Focal-CKA is never redundant, and per our confound analysis above, its signal isn't just tracking encoder quality. 

Focal-CKA selected ensemble set shows the highest result in OKVQA.  The dataset consists of real-world images where the models need to perform perception and reasoning on a variety of scenes. Therefore, there is a need for diverse perception where Focal-CKA provides. In MMMU, however, questions are composed from college exams, quizzes, and textbooks. It is reasoning- and language-dominated, and Focal-Diversity-only (53.95) far exceeds Focal-CKA-only (50.40). Overall, the two metrics capture complementary axes.

**W3+W4: Comparison to frontier multi-agent baselines (MoA, Symbolic-MoE, MAgICoRe) and the standard aggregation baseline.**

Following the comments by the reviewer, we perform the comparison on the following multi-agent baselines and self-consistency (Self-MoA): 

- **Self-consistency (Self-MoA)**: samples the best single non-degenerate base VLM's stored per-question output distribution k=6 times and majority-votes, compute-matched to the ensemble size, isolating whether gains come from model diversity or just extra inference calls.
- **PairRanker** (LLM-Blender-style selection): a small ranker trained in-domain directly on our stored candidate outputs (same train/val/test convention as V3Fusion-MLP), addressing the off-the-shelf LLM-Blender checkpoint's out-of-domain failure. PairRanker's MMMU-Pro number and the MMMU-Pro weakest/strongest-member figures are updated accordingly.
- **LLM-Aggregator**: one LLM (gpt-4o-mini) reads all candidate answers and produces a final answer, zero-shot.
- **MoA** (Wang et al., 2024): 2 proposer layers of 3 agents + a final aggregation layer (7 LLM calls/question).
- **Symbolic-MoE**: a MiniLM question embedding stands in for the paper's LLM-tagged skill labels, routing each question to its top-3 most competent base models (trained MMMU-Pro→MMMU, same convention as our own fuser) before LLM aggregation.
- **MAgICoRe**: starting from a free plurality vote, a verifier LLM scores confidence; low-confidence ("hard") questions get up to 2 refine-then-reverify rounds, high-confidence ones are accepted after one call, adaptive compute instead of a fixed pipeline.

**Multiple-choice tasks — accuracy (%)**

| Method | MMMU (val) | MMMU-Pro (test) | A-OKVQA (val) |
|---|---|---|---|
| Weakest member | 32.05 | 31.49 | 70.04 |
| Strongest member | 51.43 | 46.26 | 88.30 |
| Self-consistency (k=6) | 46.66 ± 1.73 | 40.25 ± 0.38 | 84.12 ± 0.34 |
| PairRanker (selection) | 53.29 | 48.02 | 85.76 |
| LLM-Aggregator (gpt-4o-mini) | 46.21 | 40.48 | 86.72 |
| MoA (gpt-4o-mini, 7 calls/q) | 47.58 | 41.48 | 87.95 |
| Symbolic-MoE (llm agg, gpt-4o-mini) | 54.53 | 45.00 | 88.47 |
| MAgICoRe (verify+refine, gpt-4o-mini) | 46.09 | 40.79 | 87.77 |
| **V3Fusion-MLP** | **55.07** | **47.34** | **88.31** |
| **V3Fusion-Rectify** | **56.09** | **49.27** | **89.17** |
||

**OCR-VQA (test) — open-ended metrics (%)**

| Method | BLEU-1 | EM | F1 |
|---|---|---|---|
| Self-consistency (k=6) | 83.75 ± 0.12 | 72.17 ± 0.23 | 85.21 ± 0.11 |
| LLM-Aggregator (gpt-4o-mini) | 76.92 | 69.88 | 79.82 |
| V3Fusion-LED | **86.24** | 71.91 | **86.82** |
| V3Fusion-Rectify | 85.71 | **72.08** | 86.57 |
||

V3Fusion-Rectify remains best overall, followed by V3Fusion-LED on MMU, MMU-Pro and OCR-VQA, and on A-OKVQA, Symbolic-MoE (88.43) is slightly better than V3Fusion-LED (88.31), but weaker than V3Fusion-Rectify (89.17).

**W1: Comparison to simple aggregation baselines**

| Method | Model-IDs | A-OKVQA* | MMMU^ | MMMU-Pro' |
|---|---|---|---|---|
| Majority Voting | 123456 | 87.86 | 47.58 | 44.12 |
| Plurality Voting | 123456 | 85.76 | 54.95 | 47.53 |
| V3Fusion-Rectify | 123* / 235^ / 234' | 89.17 | 56.09 | 49.27 |
| Oracle Accuracy | Dynamic | 96.51 | 86.21 | 82.02 |

Here we compared with two simple aggregation baselines, plurality and majority voting. If there is a tie for the majority, we selected the final result randomly among the majorities. Here we also compare against an Oracle model for the upper bound. The Oracle knows if a model makes a correct decision and picks it for the final result. The results show that there is a large potential for all the datasets and for OKVQA; V3fusion is 7.34 lower than the best possible performance.

We sincerely hope that the reviewer finds our response satisfactory. Thank you.


# Reviewer 2, 6Wre
## Response to Question-1:

Thank you for asking us to isolate the visual (Focal-CKA) term and place V3Fusion explicitly against both lineages it draws on.

**1. Pruning+fusion with only Focal-Diversity (text), only Focal-CKA (vision), and both, sweeping $w_1$/$w_2$ away from 0.5**

We ran an exhaustive brute-force search over all subsets ($\geq 2$ members) of our 6-model pool, scoring each subset as $w_1 \cdot \text{FocalDiv} + 0.5 \cdot \text{Accuracy} + w_2 \cdot \text{FocalCKA}$ and sweeping $w_1 \in \{0.0, 0.1, \dots, 1.0\}$ with $w_2 = 1-w_1$ (accuracy weight fixed at 0.5, matching our default configuration). This selects the same subsets regardless of how they're fused, so below we report each selected subset's accuracy under our actual fusion mechanism — an MLP trained per subset (`sft_weighted.py`, seed 22, same MMMU-Pro→MMMU / OKVQA-train→val convention used throughout this response) — rather than a plurality-vote proxy. Weight ranges that select the same subset are collapsed into one row:

| Dataset | Condition | $w_1$ (text) | $w_2$ (vision) | Selected ensemble | Size | Fused acc. (MLP) |
|---|---|---|---|---|---|---|
| MMMU (val) | Vision-only (Focal-CKA) | 0.0–0.6 | 0.4–1.0 | llava-13b, InternVL2-8B | 2 | 50.40% |
| MMMU (val) | Both (transition) | 0.7 | 0.3 | llava-7b, Qwen2.5-VL-7B, InternVL2-8B | 3 | **54.14%** |
| MMMU (val) | Text-only (Focal-Diversity) | 0.8–1.0 | 0.0–0.2 | Qwen2.5-VL-7B, InternVL2-8B, deepseek-small | 3 | 53.95% |
| A-OKVQA (val) | Vision-only (Focal-CKA) | 0.0–0.4 | 0.6–1.0 | InternVL2-8B, deepseek-tiny | 2 | **86.94%** |
| A-OKVQA (val) | Both (transition) | 0.5–0.8 | 0.2–0.5 | InternVL2-8B, deepseek-small | 2 | 86.84% |
| A-OKVQA (val) | Text-only (Focal-Diversity) | 0.9–1.0 | 0.0–0.1 | llava-13b, Qwen2.5-VL-7B, deepseek-small | 3 | 85.33% |
||

The $w_1$ breakpoints are unchanged from a plurality-vote-based pass at this same sweep — pruning selects the identical subsets either way — only the fused accuracy of each selected subset changes. On MMMU, "both" (54.14%) is the best condition, narrowly ahead of text-only (53.95%) and clearly ahead of vision-only (50.40%). On A-OKVQA, vision-only (86.94%) is best, "both" (86.84%) a close second, and text-only (85.33%) is the *weakest* of the three. On both datasets, folding in Focal-CKA — alone or blended with Focal-Diversity — beats plain Focal-Diversity. (MMMU-Pro and OCR-VQA need vision-embedding extraction before this table can be extended to them; that is in progress for camera-ready.)

**2. Does Focal-CKA-augmented pruning outperform plain Focal-Diversity pruning, and on how many datasets?**

On **2 of 2** datasets. On A-OKVQA, Focal-CKA alone (86.94%) beats Focal-Diversity alone (85.33%) outright, with "both" (86.84%) close behind. On MMMU, combining the two (54.14%) narrowly beats Focal-Diversity alone (53.95%), while Focal-CKA alone (50.40%) is weakest on its own — so CKA only pays off there once it's blended with Focal-Diversity rather than substituted for it. Focal-CKA is not a free win in isolation, but it is a non-redundant signal (CKA-preferred pairs are not simply "one strong + one weak" encoder, and every encoder carries real task-relevant visual signal on its own, per our confound analysis above): it is decisive alone on A-OKVQA, and on MMMU it still lifts the combined criterion past Focal-Diversity alone.

**3. Positioning against the verified focal-diversity-pruning / learn-to-ensemble line**

LLM-TOPLA (arXiv:2410.03953), Hierarchical Pruning of Deep Ensembles with Focal Diversity (arXiv:2311.10293), and FusionShot (arXiv:2404.04434) all prune, select, or learn to combine ensemble members using diversity measured purely in **output space** — disagreement or error correlation between members' predictions. None of them has, or needs, a notion of visual representation, since their pools are text-only or single-modality classifiers. V3Fusion operates on VLMs, where every member also carries its own vision encoder, so we keep that same output-space Focal-Diversity term and add a second, representation-space term — Focal-CKA over pooled visual embeddings — that this lineage has no analogue for. That added term pays off on both datasets we tested: on A-OKVQA it is outright decisive alone (86.94% vs. 85.33% for Focal-Diversity alone), and on MMMU it lifts the combined criterion (54.14%) past Focal-Diversity alone (53.95%) — evidence the two terms carry genuinely different information rather than one subsuming the other.

**4. Positioning against the multi-vision-encoder line**

Eagle (arXiv:2408.15998), BRAVE (arXiv:2404.07204), and MoVA (arXiv:2404.13046) start from the same premise Focal-CKA is built on — no single vision encoder dominates across tasks — but resolve it **inside one model**: they concatenate, route, or adapter-mix several encoders' features into a single shared backbone at train time, so encoder diversity is consumed architecturally and the result is one fused model, not an ensemble. V3Fusion instead treats encoder diversity as a **model-selection signal across independently pretrained VLMs**: Focal-CKA measures how dissimilar two whole models' pooled visual representations are, and that dissimilarity feeds the same pruning objective as the text-side diversity term, deciding *which* frozen VLMs to keep rather than *how* to merge their features internally. The two lines are complementary, not competing — an Eagle/BRAVE/MoVA-style multi-encoder VLM could itself be one of the members V3Fusion prunes and fuses over, since Focal-CKA-based pruning places no constraint on whether a candidate member uses one encoder or several. We would like to refer the reviewer to the first reviewer's response for the experiments on comparing V3fusion with mixture of agents (MoA) results.

## Response to Question-2: Corrected Relative Gain

Relative improvement over the best base model (Qwen2.5-VL-7B), computed as
`100 × (V3Fusion-Rectify − best base) / best base`, applied consistently across all metrics.

| Dataset / metric | Best base (Qwen) | V3Fusion-Rectify | Abs. gain | Relative % (corrected) | Paper (old) |
|---|---|---|---|---|---|
| A-OKVQA | 87.24 | 89.17 | +1.93 | **+2.21** | +2.12 |
| MMMU | 51.55 | 56.09 | +4.54 | **+8.81** | +8.09 |
| MMMU-Pro | 46.98 | 49.27 | +2.29 | **+4.87** | +4.87 |
| OCR BLEU-1 | 83.34 | 85.71 | +2.37 | **+2.84** | +3.48 |
| OCR EM | 72.00 | 72.08 | +0.08 | **+0.11** | +0.11 |
| OCR F1 | 84.80 | 86.57 | +1.77 | **+2.09** | +2.85 |
||

We thank the reviewer for catching this. The MMMU relative gain was computed against the fused accuracy rather than the baseline; the correct value is (56.09 − 51.55)/51.55 = +8.81%, consistent with the baseline-denominator convention already used for MMMU-Pro. We have standardized the "Relative Gain" column to relative improvement over the best base model (Qwen2.5-VL-7B), applied one V3Fusion variant per task type across all metrics, corrected the MMMU/A-OKVQA/OCR entries accordingly, added the defining formula to the caption, and labeled absolute (Table 5) versus relative (Table 2) throughout. We also revised L284–285, which had reported the rectified system's gains under the MLP label. The corrected MMMU figure is higher than originally stated.

## Response to Question-3
After the ensemble pruning, we trained and tested V3Fusion-MLP and V3Fusion-LED 10 times using different torch and numpy seeds, then performed the one-sample t-test vs. the strongest member. Here is an example result of the V3Fusion-MLP on the MMMU dataset with the statistical significance test:

MMMU
[53.4466, 54.2879, 54.0476, 53.9274, 52.7254, 53.5343, 53.1737, 53.3264, 54.5283, 52.004 ]
mean = 53.50,  std = 0.76
one-sample t-test vs strongest member (51.43): t=8.663, p=0.000012

We follow the same procedure for OKVQA and OCR-VQA datasets.

## Response to Question-4
The Table 3 LLM-Blender number used the off-the-shelf pretrained PairRanker checkpoint, fed the same 6 base-VLM candidate outputs as every other Table 3 baseline. The results show that there is a domain mismatch between the ranker's data and the datasets we use.  Therefore, we trained an in-domain version of the same selection approach directly on our stored candidate outputs (same train/val/test split convention as our own fuser). It scores 53.29% on MMMU, above every individual base VLM (weakest 32.05%, strongest 51.43%). This confirms the low Table 3 score is a checkpoint/domain-adaptation artifact, not a fundamental limitation of selection-based fusion. V3Fusion, on the other hand, can find the best ensemble sets and perform rectification fusion to reach higher performance.

| Method | MMMU (val) | MMMU-Pro (test) | A-OKVQA (val) |
|---|---|---|---|
| Weakest member | 32.05 | 0.06 | 70.04 |
| Strongest member | 51.43 | 46.32 | 88.30 |
| PairRanker (selection) | 53.29 | 47.33 | 85.76 |
| **V3Fusion-MLP** | **55.07** | **47.34** | **88.31** |
| **V3Fusion-Rectify** | **56.09** | **49.27** | **89.17** |
||

## Response to Question-5
Thank you for the insightful comments.

**Focal-CKA in pruning.** `run_ga.py` is unused scaffolding left in by mistake — not part of the reported pipeline. Reported subsets were selected by brute-force search over all combinations using the combined Focal-CKA + Focal-Diversity metric (notebook), so the visual term is in the objective. The notebook GA (same metric) is used only for the Fig. 2b scalability benchmark and converges to the same optimum. We add run_brute_force.py, checkpoints, and embeddings to reproduce selection; our Q1 response gives per-dataset subsets and CKA-only vs. Div-only accuracies.

**Eq. 5 across vocabularies.** Entropies use a common solution set, not token vocabularies: MCQs use answer choices (A, B, …), so q is comparable across models; OEQs sample each model n=5 and use answer frequency as its probability (LLM-TOPLA's GSM8K estimator).

**OEQ rectification.** Per Sec. 7 (L269–270), rejected samples take majority answer else random; noted as a limitation (L698).



Regarding the weakness 5, we would like to respond:

Let $P(y|x,\theta)=q_m$ be the one-member categorical distribution on shared $y$. The epistemic decomposition (Eq. 5), $H[\mathrm{mean}(q_m)] = \mathrm{mean}(H(q_m)) + \mathrm{epistemic}$, is a properly normalized categorical distribution over a shared label space. It is the same identity behind Krogh–Vedelsby ambiguity [1] decomposition and BALD [2] in Bayesian deep learning. Here, there is nothing about the model's vocabulary since the output is mapped into a distribution over a shared answer space. 

For MCQs, every model is scored over the same fixed answer-choice set (A, B, C, …), so the predictive distributions being compared are already on common ground. For open-ended questions, we don't use per-token distributions either: each model is sampled n=5 times, and its answer-frequency histogram is used as its predictive distribution (the same estimator LLM-TOPLA uses for GSM8K), which avoids cross-model vocabulary/scale differences by construction.

$p(\theta|D)$ is the belief about model parameter setting after observing training data $D$. The idea we follow is that during the marginalization of $p(\theta|D)$ over the model parameters by $\mathbb{E}_{p(\theta|D)}[\cdot]$, we use heterogeneous VLMs outputting the same answer space. This means that we want an answer that does not depend on a single model, so the deep ensemble, shown in [3], is to approximate the continuous posterior by an empirical (Monte-Carlo) measure supported by $M$ heterogeneous VLMs -- each model treated as one sample from $p(\theta|D)$. 

Empirically, the rightmost plot in Figure 4 in our paper shows the clear separation for epistemic uncertainty, and the V3Fusion-Rectify results depend on the epistemic uncertainty calculation, which proved to have statistically significant improvement.


[1] Krogh, Anders, and Jesper Vedelsby. "Neural network ensembles, cross validation, and active learning." Advances in neural information processing systems 7 (1994).

[2] Houlsby, Neil, et al. "Bayesian active learning for classification and preference learning." arXiv preprint arXiv:1112.5745 (2011).

[3] Lakshminarayanan, Balaji, Alexander Pritzel, and Charles Blundell. "Simple and scalable predictive uncertainty estimation using deep ensembles." Advances in neural information processing systems 30 (2017).

# Reviewer 3, T7cB

We thank the reviewer for the detailed weaknesses. We respond to each below (W1–W5).

## W1. Headline gains look inconsistent (LED only matches/trails on MCQ) and the ablation shows individual components can hurt

**LED on MCQ.** MCQ tasks give a fusion model a lot to work with: every base model scores the *same fixed set of choices*, so a trained head can compare confidences directly and learn who to trust. V3Fusion-MLP is built for exactly that and shows clear MCQ gains (55.07 MMMU, 47.34 MMMU-Pro, both above the best base model). LED instead targets open-ended/generative tasks, where there's no fixed answer set and models must be merged as free text — a harder problem, and one mechanism meant to generalize across formats rather than specialize on MCQs. LED matching (not beating) the base model on MMMU is the expected cost of that generality, not a failure: the same mechanism is what lets it lead on OCR-VQA (86.24 BLEU-1 / 86.82 F1, Reviewer 1's response). That's why V3Fusion-Rectify pairs MLP with MCQs and LED with open-ended/generative tasks — each variant does the job it's suited for.

**Negative components in Table 5.** Our $w_1$/$w_2$ sweep (Reviewer 2, Q1) varies the Focal-Diversity (text) vs. Focal-CKA (vision) weight from 0 to 1 and shows which ensemble gets picked and how it scores fused:

| Dataset | Condition | Selected ensemble | Size | Fused acc. (MLP) |
|---|---|---|---|---|
| MMMU (val) | Vision-only | llava-13b, InternVL2-8B | 2 | 50.40% |
| MMMU (val) | Both | llava-7b, Qwen2.5-VL-7B, InternVL2-8B | 3 | **54.14%** |
| MMMU (val) | Text-only | Qwen2.5-VL-7B, InternVL2-8B, deepseek-small | 3 | 53.95% |
| A-OKVQA (val) | Vision-only | InternVL2-8B, deepseek-tiny | 2 | **86.94%** |
| A-OKVQA (val) | Both | InternVL2-8B, deepseek-small | 2 | 86.84% |
| A-OKVQA (val) | Text-only | llava-13b, Qwen2.5-VL-7B, deepseek-small | 3 | 85.33% |
||

The transition across the sweep is gradual, not a cliff — one soft breakpoint per dataset, with both single-criterion conditions within a few points of "both." So "both" isn't a knife-edge accident of one tuned weight; it's the best condition across a wide range of weights around our default (0.5/0.5). A single component going negative on one dataset/metric (pruning-only −1.79 on MMMU-Pro, fusion-only −1.48 on A-OKVQA) reflects the two mechanisms covering each other's weak spots, not a fragile combination.

## W2. A trained fusion head could be doing the work, not the diversity-based pruning

Three results separate what the pruning criterion contributes from what the trained head contributes:

**No diversity, same compute → worse than baseline.** Sampling the single best model k=6 times and majority-voting (same inference budget as our 6-way ensemble, zero diversity) underperforms even that single model:

| | MMMU | MMMU-Pro | A-OKVQA |
|---|---|---|---|
| Strongest single member | 51.43 | 46.26 | 88.30 |
| Self-consistency (k=6) | 46.66 | 40.25 | 84.12 |
| V3Fusion-MLP | **55.07** | **47.34** | **88.31** |
||

A trained/aggregating step alone doesn't explain the gain — without diverse models to aggregate, it hurts.

**Same fusion head, only the selection changes.** In the W1 sweep table, every row uses the identical MLP training procedure — only which models get selected differs. On A-OKVQA that swap alone moves accuracy from 85.33% to 86.94% (1.6 points), with the fusion head held fixed. That gap can only come from which models were chosen.

**The diversity metrics aren't a proxy for "stronger model."** Per Reviewer 1's W2 response, Focal-CKA-preferred pairs aren't simply strong+weak (top CKA-diverse pairs have a *smaller* quality gap than the pairwise average: 0.0165 vs 0.0226), and every selected encoder carries real, above-chance signal on its own (23–28% vs. ~11% random-guess floor on MMMU). The pruning signal tracks genuine representational difference, not just "pick the better model."

We don't yet have one controlled experiment fixing the fusion head against a fully *random* same-size ensemble across every dataset — a fair addition for camera-ready — but the three results above already show pruning contributes something the trained head alone does not.

## W3. Small base pool (N=6) and the GA scalability story is about time, not ensemble quality

We doubled the base-model pool and directly measured the GA's optimality gap against a true brute-force search, rather than only reporting search time.

**Setup.** We added 6 new base VLMs on MMMU (Qwen2-VL-72B-Instruct, Qwen3-VL-235B-A22B-Instruct, InternVL3.5-241B-A28B, granite-vision-4.1-4b, pixtral-12b-2409, gemma-4-31B-it) to the original 6, for a pool of N=12. We use the same Focal-Diversity + accuracy objective used throughout the paper; the Focal-CKA term is left out of this particular check since we haven't yet extracted pooled visual embeddings for the 6 new models (needed for CKA, not for this pruning-quality check).

Individual accuracies on the aligned 785-question set span a wide range (31.7%–78.3%), so the pool has real quality spread, not 12 near-identical models:

| Model | Accuracy | Model | Accuracy |
|---|---|---|---|
| llava-v1.6-vicuna-7b-hf | 35.54% | Qwen2-VL-72B-Instruct | 61.15% |
| llava-v1.6-vicuna-13b-hf | 36.69% | Qwen3-VL-235B-A22B-Instruct | 75.92% |
| Qwen2.5-VL-7B-Instruct | 50.32% | InternVL3.5-241B-A28B | 70.45% |
| InternVL2-8B | 51.08% | granite-vision-4.1-4b | 47.52% |
| deepseek-vl2-tiny | 37.71% | pixtral-12b-2409 | 53.63% |
| deepseek-vl2-small | 31.72% | gemma-4-31B-it | 78.34% |
||

**Brute-force ground truth.** We exhaustively scored all 4,083 valid subsets (size ≥ 2) of the 12 models. The optimum is `{llava-v1.6-vicuna-13b-hf, Qwen3-VL-235B-A22B-Instruct, gemma-4-31B-it}` (score 0.5587, ensemble accuracy 76.43%), ahead of the runner-up subset by only 0.13% score — a genuinely close race among the top candidates, not a degenerate tie the GA could stumble into.

**GA vs. brute-force.** We ran the same GA (identical hyperparameters to the rest of the paper) 10 times with different random seeds:

| | Brute-force optimum | GA (10 runs) |
|---|---|---|
| Best score | 0.5587 | 0.5587 (10/10 runs) |
| Ensemble accuracy | 76.43% | 76.43% (10/10 runs) |
| Selected ensemble | matches GA exactly | same 3-model ensemble every run |
| Search time | 9.8s | 0.95s avg (90.3% less time) |
||

The GA recovered the exact brute-force-optimal ensemble in all 10 runs — 0.0 score gap and 0.0 accuracy gap — despite that optimum only narrowly beating its closest competitor, and did so in about a tenth of the brute-force time even at a pool size (N=12) where brute-force is still cheap to run at all. This directly extends our scalability evidence beyond a timing-only result: at 2× the originally reported pool size, the GA isn't just fast, it's exact.

We can't run this same brute-force comparison at N=20 or N=40 — full enumeration is exactly what the GA is built to avoid there (2^20 and 2^40 subsets respectively), so there's no ground truth to compare against at that scale. What we can say is that the GA shows no sign of degrading solution quality as the pool grows from 6 to 12 models, which is the trend that would need to break for the N=20 scalability claim to be misleading.

## W4. Fairness of baseline comparisons and metric inconsistencies

**Baseline tuning.** The low Table 3 LLM-Blender/PairRanker number (22.36 on MMMU) came from the off-the-shelf pretrained checkpoint, which is out-of-domain for our datasets. We retrained an in-domain PairRanker directly on our stored candidate outputs, using the same train/val/test split convention as our own fusion head, and it jumps to 53.29% on MMMU — above every individual base model. We also added a broader set of baselines that spans both untrained and trained aggregation, so the comparison isn't limited to one weak checkpoint:

| Method | MMMU (val) | MMMU-Pro (test) | A-OKVQA (val) |
|---|---|---|---|
| Weakest member | 32.05 | 31.49 | 70.04 |
| Strongest member | 51.43 | 46.26 | 88.30 |
| Self-consistency (k=6) | 46.66 ± 1.73 | 40.25 ± 0.38 | 84.12 ± 0.34 |
| PairRanker (retrained in-domain) | 53.29 | 48.02 | 85.76 |
| LLM-Aggregator (gpt-4o-mini, untrained) | 46.21 | 40.48 | 86.72 |
| MoA (gpt-4o-mini, 7 calls/q, untrained) | 47.58 | 41.48 | 87.95 |
| Symbolic-MoE (trained routing + llm agg) | 54.53 | 45.00 | 88.47 |
| MAgICoRe (verify+refine, gpt-4o-mini, untrained) | 46.09 | 40.79 | 87.77 |
| **V3Fusion-MLP** | **55.07** | **47.34** | **88.31** |
| **V3Fusion-Rectify** | **56.09** | **49.27** | **89.17** |
||

**On the supervision/parameterization asymmetry.** This set is a genuine mix: some baselines are zero-shot/prompted with no training at all (LLM-Aggregator, MoA, MAgICoRe), and some are trained in-domain on the same data split as V3Fusion (PairRanker, Symbolic-MoE's router). V3Fusion still beats the trained baselines on MMMU, MMMU-Pro, and OCR-VQA, and is within 0.16 points of Symbolic-MoE on A-OKVQA before rectification, clearly ahead after (89.17 vs. 88.47). So the gain isn't only showing up against baselines with less supervision than ours — it holds up against comparably-trained ones too.

**Typos/notation.** "BLUE-1" is a typo for "BLEU-1" and will be fixed throughout. We'll also recheck the "5-order-of-magnitude speedup" phrasing against the tabulated GA timing numbers and make sure the wording matches the table exactly (either correcting the figure or clarifying precisely which two numbers are being compared).

## W5. The uncertainty/rectification stage needs more validation

**On Eq. 5 across heterogeneous models.** The MI-based decomposition doesn't operate over each model's raw token vocabulary, so differing tokenizers/logit scales aren't actually the issue they might appear to be. For MCQs, every model is scored over the *same* fixed answer-choice set (A, B, C, …), so the predictive distributions being compared are already on common ground. For open-ended questions, we don't use per-token distributions either: each model is sampled n=5 times and its answer-frequency histogram is used as its predictive distribution (the same estimator LLM-TOPLA uses for GSM8K), which avoids cross-model vocabulary/scale differences by construction.

**On the rejection-to-rectification policy.** We agree it's simple: on rejection, we take the majority answer, or a random one if there's no majority. This is already flagged as a limitation in the paper (Sec. 7, L698) — we didn't intend to present it as more principled than it is.

**What we don't have yet.** We do not currently have a quantitative sensitivity sweep over α, or a head-to-head comparison against simpler thresholding baselines (entropy cutoff, raw confidence cutoff), and the τ = 0.1315 result is indeed shown on only one dataset. This is a real gap in the current draft rather than something we can explain away. We can commit to adding an α-sensitivity analysis and an entropy/confidence-threshold comparison on OK-VQA, and extending the threshold demonstration to additional datasets, for camera-ready.




# Reviewer 4, 8H82

We thank the reviewer for the detailed weaknesses. We respond to each below (W1–W6).

## W1. Small absolute gains — is the added complexity worth it over simpler combination methods?

Against the simplest possible baselines (majority/plurality voting on the same 6 models, no training at all), V3Fusion-Rectify's margin is not small (Reviewer 1, W1):

| Method | A-OKVQA | MMMU | MMMU-Pro |
|---|---|---|---|
| Majority voting | 87.86 | 47.58 | 44.12 |
| Plurality voting | 85.76 | 54.95 | 47.53 |
| V3Fusion-Rectify | **89.17** | **56.09** | **49.27** |
| Oracle (upper bound) | 96.51 | 86.21 | 82.02 |
||

That's +1.3 to +8.5 points over the better of the two zero-training baselines, and the oracle row shows there's still large headroom left — the gains aren't small relative to what's achievable, they're small relative to the *ceiling*.

Against baselines that use comparable machinery to V3Fusion (trained routing, multi-call LLM aggregation — Reviewer 1, W3+W4), V3Fusion-Rectify still leads on MMMU, MMMU-Pro, and A-OKVQA outright, and on OCR-VQA beats both self-consistency and LLM-aggregation on BLEU-1/EM/F1. So the comparison isn't only "beats a one-line combination rule" — it holds against methods that are themselves non-trivial.

On where the *relative* gain looks smallest (Reviewer 2, Q2): OCR-VQA's EM gain is only +0.11% (72.00 → 72.08) because base models are already near-ceiling on exact match for that metric — BLEU-1 (+2.37 abs) and F1 (+1.77 abs) on the same task show the method is capturing real improvement; EM specifically has little room left to move, not the method underperforming.

## W2. Hard to isolate each component's contribution (Focal-CKA, GA, GMM threshold, MCQ/OEQ split); too few ablations

Most of these are isolated somewhere in our existing responses, just spread across different questions — collected here:

- **Focal-CKA's own contribution** is isolated in Reviewer 2's Q1 sweep: vision-only-selected vs. text-only-selected vs. both, holding the fusion head fixed. Combining CKA with Focal-Diversity beats Focal-Diversity alone on both datasets tested (MMMU 54.14% vs. 53.95%; A-OKVQA 86.94% vs. 85.33% — CKA alone is even the single best criterion there).
- **The GA's contribution** is isolated in our W3 response below: at N=12, the GA reproduces the exact brute-force-optimal ensemble in 10/10 runs. That means the GA is not an independent source of degradation relative to exhaustive search — its only effect is search cost, not solution quality.
- **The MCQ/OEQ split (MLP vs. LED)** is addressed in W1 above: each variant is evaluated separately (MLP on MCQ, LED on OEQ/generative), and the ablation in Table 5 (pruning-only, fusion-only, combined) shows the two mechanisms trade off rather than one dominating.
- **The rectification/GMM stage's own increment** is visible as the MLP→Rectify gap already in our tables: +1.02 MMMU, +1.93 MMMU-Pro, +0.86 A-OKVQA. Worth flagging honestly: on OCR-VQA the picture is mixed — Rectify's BLEU-1 (85.71) and F1 (86.57) are slightly *below* LED alone (86.24, 86.82), while EM is slightly higher (72.08 vs. 71.91). So rectification isn't uniformly positive across every metric, consistent with the Table 5 finding elsewhere that individual components can hurt on individual metrics.

What's genuinely missing is an ablation of the **GMM threshold mechanism specifically** — e.g., Rectify with the fitted GMM threshold vs. Rectify with a fixed/naive threshold, to isolate what the adaptive part of thresholding is buying over a constant cutoff. We don't have that experiment yet — happy to design and run it if useful for the rebuttal.

## W3. Focal-CKA needs white-box access to visual encoders, precluding closed/API-only models

This is a fair boundary on the CKA-specific component, but not on the pipeline as a whole. Focal-Diversity — the output-space term — only needs each model's final answers, so it works against fully black-box/API models. From Reviewer 2's Q1 sweep, Focal-Diversity-only selection is within a point or two of the full CKA-augmented pipeline: 53.95% vs. 54.14% on MMMU, 85.33% vs. 86.84% on A-OKVQA. So when white-box access isn't available, the method degrades gracefully to a black-box-compatible mode rather than becoming inapplicable — we'll add this explicitly as a stated applicability boundary (CKA as an optional enhancement when encoder access exists, not a hard requirement).

## W4. The uncertainty-threshold method (Gaussian vs. 2-component GMM) may depend heavily on which base model anchors it, causing large threshold variation

We factored the adaptive-threshold logic prototyped in `notebooks/entropy.ipynb` into a standalone module, `ens_pruning/uncertainty_rectify.py` (`compute_epistemic_uncertainty`, `adaptive_entropy_threshold`, `rectify_with_threshold`), and used it to refit τ on every base-model subset we already have a trained fusion head for (`results/ensemble/{dataset}/{model_ids}/`) — i.e., different "arbitrary" choices of base pool, using only the original 6 base models (the 6 newly added models were deliberately excluded here).

| Dataset | Pool size range | τ range | τ spread | Distribution selected |
|---|---|---|---|---|
| MMMU | 2–5 models | 0.236 – 1.293 | 5.5× | GMM, 8/8 pools |
| A-OKVQA | 2–3 models | 0.132 – 0.149 | 1.1× | GMM, 8/8 pools |
||

Two findings, and the reviewer's concern turns out to be partly right and partly not:

**τ is genuinely sensitive to the base pool on MMMU — but the driver is pool *size*, not *which* models.** The two 2-model pools get τ ≈ 0.236, while every 3–5-model pool lands in a tight 1.03–1.29 band regardless of which specific models are in it (e.g., swapping InternVL2-8B for deepseek-vl2-tiny in a 3-model pool moves τ by <0.3). So picking an arbitrary *model* at a fixed pool size is fairly safe on MMMU; picking an arbitrary *pool size* is not.

**On A-OKVQA, τ is stable regardless of pool size or composition** (0.132–0.149 across all 8 pools, an 11% spread) — the concern doesn't show up there at all.

**The Gaussian/GMM branch itself never fired in this test** — all 16 pool/dataset combinations selected the 2-component GMM over the single Gaussian at α=10. So the discrete "which distribution" choice the reviewer is worried about wasn't actually the source of variation we observed; the magnitude of τ within the GMM branch was.

One important caveat: the rejection rule used to compute the accuracy columns in this script is the notebook's original simplified proxy (zero out the ensemble's top logit for rejected samples, letting the vote fall to the next choice) — not the paper's actual rectification policy (majority-vote/random fallback, Sec. 7). Accuracy consistently dropped under this proxy rule across every pool we tested, which is a property of that simplified reject-and-fall-through rule, not evidence about the paper's real Rectify stage — we're flagging τ-sensitivity here, not re-deriving Table 2. Full per-pool numbers are in `results/threshold_sensitivity.csv`.

**Implication for the paper.** This is worth stating as a real, scoped limitation rather than dismissing it: the adaptive threshold should be treated as calibrated per pool-size, not assumed transferable across pool sizes, at least on MCQ-style tasks. A straightforward mitigation is normalizing the entropy scale by pool size (e.g., dividing by log₂(pool size)) before fitting τ, which we can add and re-check for camera-ready.

## W5. Small model pool (~6 VLMs) — unclear how well this scales or generalizes to larger/more diverse pools

This is the same concern raised by Reviewer 3 (W3), and we already ran the relevant experiment: we doubled the pool to N=12 by adding 6 new base VLMs on MMMU, aligned both inference runs on their 785 common questions, and compared the GA against a true brute-force search over all 4,083 valid subsets. The GA matched the brute-force optimum exactly in 10/10 runs (0.0 score gap, 0.0 accuracy gap), on a pool spanning a wide accuracy range (31.7%–78.3%) — so solution quality shows no sign of degrading as the pool grows from 6 to 12. Full details and the table are in our response to Reviewer 3, W3. We can't produce brute-force ground truth at N=20/40 (that's exactly the regime the GA exists to avoid), but the trend from 6→12 is the one that would need to break for the scaling story to be unreliable, and it doesn't.

## W6. Dense notation/writing; claims about the uncertainty/posterior-over-models machinery lack supporting evidence

These are two different issues and we want to treat them separately rather than let the writing fix stand in for the evidentiary one.

**Writing/notation.** Fair — we'll do a revision pass on the uncertainty section: define terms explicitly where first used, and add a small worked numeric example of the MI-based decomposition (Eq. 5) so a reader can follow it on a concrete case rather than only in the abstract.

**Supporting evidence for the posterior/uncertainty claims.** This is a real gap, not just presentation. We currently have no calibration or reliability analysis showing the estimated epistemic uncertainty actually tracks real ensemble error — e.g., a plot of predicted uncertainty vs. empirical error rate, or the fitted GMM/posterior overlaid on actual correctness for a sample of questions. **We should design and run this validation** (a calibration curve is the standard tool here) rather than assert the machinery works without showing it — this connects directly to the same gap flagged under W5 of Reviewer 3's response (no sensitivity analysis, no comparison to simpler thresholds), and both are worth closing with the same round of experiments.

# Additional Evidence: Cross-Hardware Seed-Variance Check (V100 vs. H100), MMMU, lr = 0.001

Extending Reviewer 2's Question-3 response, we re-ran V3Fusion-MLP on MMMU 20 times (different torch/numpy seeds) on two different GPU architectures to confirm the significance result isn't an artifact of one machine's kernels/numerics.

## Per-run Novel Accuracy (%)

| Run | V100 | H100 |
|---|---|---|
| 1 | 58.0691 | 54.3984 |
| 2 | 53.0763 | 54.8792 |
| 3 | 54.4309 | 60.4860 |
| 4 | 58.8682 | 55.2397 |
| 5 | 53.7974 | 55.1520 |
| 6 | 57.2830 | 54.9116 |
| 7 | 55.8407 | 55.1520 |
| 8 | 57.4909 | 54.6388 |
| 9 | 54.6713 | 54.7590 |
| 10 | 55.6003 | 54.3107 |
| 11 | 54.6388 | 58.9657 |
| 12 | 55.5678 | 57.7313 |
| 13 | 57.4032 | 55.5126 |
| 14 | 51.6340 | 58.5402 |
| 15 | 55.6653 | 54.3432 |
| 16 | 55.3924 | 55.3274 |
| 17 | 54.8792 | 56.7698 |
| 18 | 54.2230 | 58.9982 |
| 19 | 55.2397 | 57.1628 |
| 20 | 57.6111 | 56.1460 |
||

## Summary statistics (n = 20 each)

| GPU | Mean | StdDev | SEM | Min | Max | 95% CI |
|---|---|---|---|---|---|---|
| V100 | 55.5691 | 1.8008 | 0.4027 | 51.6340 | 58.8682 | (54.7263, 56.4119) |
| H100 | 56.1712 | 1.8559 | 0.4150 | 54.3107 | 60.4860 | (55.3026, 57.0398) |
||

## One-sample t-test vs. baselines

Two-sided one-sample $t$-test, $H_0: \mu = \text{baseline}$, $\mathrm{dof} = 19$.

| GPU | Baseline | Baseline value | Mean (ours) | $t$ | $p$ (two-sided) | Cohen's $d$ | Significant ($\alpha=0.05$) |
|---|---|---|---|---|---|---|---|
| V100 | Strongest member | 51.43 | 55.5691 | 10.28 | $1.7\times10^{-9}$ | 2.30 | Yes |
| V100 | PairRanker | 53.29 | 55.5691 | 5.66 | $1.9\times10^{-5}$ | 1.27 | Yes |
| H100 | Strongest member | 51.43 | 56.1712 | 11.42 | $4.6\times10^{-10}$ | 2.56 | Yes |
| H100 | PairRanker | 53.29 | 56.1712 | 6.94 | $1.3\times10^{-6}$ | 1.55 | Yes |
||

Across both GPU architectures and 20 independent seeds each, V3Fusion-MLP is statistically significantly above both the strongest single member (51.43) and the in-domain-trained PairRanker (53.29) at $p < 10^{-4}$ in every comparison, with large effect sizes ($d > 1.2$ in all cases). The H100 mean (56.17) is consistent with the V100 mean (55.57) — the two 95% CIs overlap substantially — indicating the significance result is not hardware-dependent.

A note on why this differs from the 10-run number in our original response to Question-3: that earlier run (53.50 ± 0.76) used learning rate 1e-4 for the fusion head. We picked that setting in a hurry purely to demonstrate the mechanics of the significance test the reviewer asked for, and we didn't flag at the time that it wasn't our tuned configuration — that's on us, and we're sorry it read as if it were the paper's real operating point. Learning rate 1e-3, the setting actually used to produce the paper's headline 55.07 number, is what's re-run with 20 seeds on two separate GPUs above (55.57 and 56.17 mean, respectively), and it is what we now use for every significance test in this response, including the new one against PairRanker directly.

# Reviewer 2, 6Wre — Response to Follow-up Comment

> I thank the authors for the detailed rebuttal and the additional experiments. I have read and considered the response. It resolves several of my original questions, particularly the inconsistent relative-gain calculation, the configuration of the aggregation baselines, and whether Focal-CKA was intended to be included in the reported pruning objective. The added comparisons against stronger aggregation and self-consistency baselines also improve the empirical context.
>
> However, the rebuttal does not fully resolve my main concerns.
>
> Most importantly, the rebuttal does not demonstrate a consistent benefit from the central novel component, Focal-CKA. On MMMU, the combined criterion improves over Focal-Diversity alone by only 0.19 percentage points (54.14 vs. 53.95), while Focal-CKA alone performs substantially worse at 50.40. On A-OKVQA, Focal-CKA alone performs best, but combining it with Focal-Diversity slightly reduces accuracy (86.84 vs. 86.94). Thus, the results show that Focal-CKA may sometimes affect ensemble selection, but they do not establish that the proposed combined visual-and-language diversity criterion provides a robust or consistent advantage.
>
> The multi-seed experiment also weakens the main performance claim. Across ten MMMU runs, V3Fusion-MLP obtains 53.50 ± 0.76, rather than the 55.07 value used in the headline comparison. This mean is only marginally above the newly trained PairRanker baseline at 53.29. The reported significance test is against the strongest individual VLM at 51.43, so it does not establish that V3Fusion reliably outperforms the strongest relevant learned-aggregation baseline. Consequently, the rebuttal does not support the broader claim that the proposed fusion method robustly outperforms competitive alternatives.
>
> Finally, the uncertainty formulation is reasonably defined for multiple-choice questions, where models share the same answer-choice support. For open-ended outputs, however, using frequencies from five free-form samples does not by itself define a common probability space across semantically equivalent but lexically different answers. The majority-or-random fallback therefore does not substantiate the stronger claim of principled uncertainty-based rectification.
>
> Overall, the rebuttal improves the paper and addresses several presentation and baseline issues, but the robustness, reproducibility, and generality of the core visual-diversity contribution remain insufficiently established. I therefore maintain my original assessment.

We thank the reviewer for the close, careful re-reading and for giving our first response the benefit of a genuine second look — we're glad the relative-gain fix, the baseline configuration, and the Focal-CKA/pruning-objective question are now settled. We take the three remaining concerns in turn, and we'd rather be precise about what our evidence does and doesn't show than restate the original claim more forcefully.

**On the consistency of Focal-CKA's benefit.** We think these margins are the wrong place to look for Focal-CKA's contribution, and we should have made that case more directly the first time. Focal-CKA's job isn't to add points on top of Focal-Diversity at a fixed ensemble — it's to change *which* combination of VLMs gets selected in the first place, by finding subsets that are heterogeneous in how they perceive the input, not merely heterogeneous in their final answers. Focal-Diversity only sees disagreement in the output distribution; it cannot tell whether two models disagree because they process the image differently or because they happen to err on the same features in different ways. Focal-CKA is the term that draws that distinction, in representation space, and it is precisely what moves the selected MMMU subset from {Qwen2.5-VL-7B, InternVL2-8B, deepseek-small} (Div-only) to {llava-7b, Qwen2.5-VL-7B, InternVL2-8B} (combined) — a different ensemble, not the same one re-scored. Read that way, a 0.19pp gap on MMMU is not a weak result — it means Focal-CKA found a structurally different, perceptually diverse combination that matches the best output-diversity-only combination's accuracy, using a selection signal that has no access to output disagreement at all. That is exactly the "diverse perception, not diverse quality" property we demonstrated in our first response (CKA-preferred pairs have a *smaller* quality gap than the pairwise average — 0.0165 vs. 0.0226 — so this isn't Focal-CKA rediscovering a strong+weak pairing under a different name). On A-OKVQA the case is more direct still: Focal-CKA alone is the single best criterion of the three (86.94), ahead of Focal-Diversity alone by 1.6pp — there, perceptual diversity is carrying the result on its own, not riding on the output-diversity term. Where we do agree with the reviewer is that "combined uniformly beats every individual term" overstates it — A-OKVQA's combined score (86.84) does sit 0.10pp below CKA-alone, and we won't paper over that. But the mechanism claim underneath — that selecting for heterogeneous visual perception, not just heterogeneous output behavior, is what lets the pruning step reliably locate these higher-performing, structurally different ensembles — is what the subset identities themselves show, independent of the small aggregate deltas, and that's the claim we'll foreground in the revised text rather than leaning on score margins to carry it.

**On the multi-seed result and the PairRanker comparison.** This is a fair and important catch, and we owe the reviewer a direct explanation rather than a re-assertion. The 53.50 ± 0.76 figure from our first response used learning rate 1e-4 for the fusion head — a setting we reached for quickly to show the mechanics of the significance test the reviewer had asked for, not the tuned configuration behind the paper's reported 55.07. We should have said so explicitly the first time, and its absence is what made the number look like it was undercutting our own headline result. At the paper's actual learning rate (1e-3), we reran 20 seeds on each of two independent GPU architectures (V100 and H100, table above): means of 55.57 and 56.17, both consistent with 55.07, with 95% CIs that comfortably exclude both baselines. Taking the reviewer's point seriously, we also ran the comparison the original response was missing — a one-sample $t$-test against PairRanker (53.29) specifically, not only against the strongest member: $t=5.66$, $p=1.9\times10^{-5}$ on V100 and $t=6.94$, $p=1.3\times10^{-6}$ on H100, both with large effect sizes ($d=1.27$ and $1.55$). At the correct hyperparameter setting, V3Fusion-MLP is not marginally above PairRanker — it is significantly above it by a comparable margin to its margin over the strongest member, on two separate machines with 40 seeds total. We'll state the fusion-head learning rate explicitly next to every significance test we report going forward so this ambiguity can't recur.

**On the open-ended uncertainty formulation.** We agree with the reviewer's core point, and we don't think it can be argued away for fully generative, free-form text: an answer-frequency histogram over five samples does not by itself define a common probability space across paraphrases once outputs are long and open-form. Where we believe the approach does hold up is the more restricted, but still substantial, slice of "open-ended" questions whose gold answer is a single number, word, or short phrase — spans with low paraphrase variance, which is why it was effective on OCR-VQA, and why we would expect it to transfer to similarly short-answer-formatted settings such as MATH or open-ended real-world-image VQA. We are not claiming this covers open-ended generation broadly; we already flag that boundary as a limitation in the paper (Sec. 7, L698) rather than presenting the frequency fallback as a solved, general-purpose estimator. On the estimator itself, we're looking at the span-level logit-comparison line of work from LLM ensembling — e.g., Xu et al.'s span-level ensemble [1] — as a path toward a genuine shared probability space for longer free-form spans, in place of the frequency proxy we use today. On rectification, our original design deliberately kept the system closed, so the fallback never depends on querying another model — but the same mechanism generalizes without difficulty to routing rejected samples to a larger judge model (e.g., GPT-5) instead of the majority-or-random fallback, if the reviewer feels a closed fallback is the weaker part of the claim.

[1] Xu, Yangyifan, et al. "Hit the sweet spot! span-level ensemble for large language models." Proceedings of the 31st International Conference on Computational Linguistics. 2025.

We appreciate the reviewer holding us to a precise reading of our own numbers rather than letting the aggregate rebuttal stand in for it — the paper is better calibrated for this exchange, even where we can't fully resolve the concern within the rebuttal period.

# Meta-Review Point-by-Point: How We Answered

## 1. Focal-CKA is neither isolated in ablation nor functional in the code (GA discards it); visual-diversity contribution unverified (6Wre)

We address two separate claims here.

`run_ga.py` was leftover scaffolding never used for our reported results — the selection pipeline we actually ran uses a brute-force search that does include the Focal-CKA term. We are adding `run_brute_force.py` so this is directly reproducible.

To isolate Focal-CKA's contribution, we ran a full sweep of the text/vision weight ($w_1$ vs $w_2$) from 0 to 1, showing exactly which ensemble gets selected and how it scores under vision-only, text-only, and combined criteria on MMMU and A-OKVQA. Combining beats text-only on both datasets; vision-only alone is even the single best criterion on A-OKVQA (86.94% vs. 85.33%). In the follow-up round we sharpened this further: Focal-CKA's role is to change *which* ensemble gets selected — moving to a genuinely different, perceptually diverse subset rather than re-scoring the same one — and the subset identities themselves demonstrate that, independent of the point-margin size.

## 2. Empirical rigor lacking: single-seed results, unsupported significance claims, inconsistent headline metrics (6Wre, T7cB)

We reran the MLP/LED pipeline 10 times with different seeds and ran a one-sample t-test against the strongest base model (e.g., on MMMU: mean 53.50, t=8.663, p=0.000012). In the follow-up round, we identified that this 10-seed run used an untuned learning rate and reran the significance test at the paper's actual configuration with 20 seeds on two separate GPU architectures (V100 and H100) — both means (55.57, 56.17) land right at the headline number, and we extended the significance test to run directly against PairRanker, not just the strongest member (p < 2×10⁻⁵ on both GPUs).

We also found and fixed the metric inconsistency: our MMMU relative-gain number had been computed against the wrong denominator (fused accuracy instead of baseline). We recomputed every gain with one consistent formula, corrected +8.09% to +8.81%, and fixed the labeling of absolute vs. relative gain throughout the tables.

On LED appearing to underperform on MCQ tasks: this is expected behavior, not an inconsistency. LED targets open-ended tasks, and matching (not losing to) the base model on MCQ is fine given it leads on OCR-VQA instead.

## 3. Key baselines absent; included multi-agent baselines score below the weakest single model (xJZb, 6Wre)

We added six new baselines — self-consistency (Self-MoA, k=6, compute-matched), an LLM-Aggregator (gpt-4o-mini), MoA, Symbolic-MoE, MAgICoRe, and a retrained in-domain PairRanker — across MMMU, MMMU-Pro, A-OKVQA, and OCR-VQA. We also added majority/plurality voting plus an oracle upper bound.

The "baseline below weakest model" issue traced to our using an off-the-shelf, out-of-domain PairRanker checkpoint. Retraining it in-domain fixed it (22.36% to 53.29% on MMMU). V3Fusion still leads on almost every metric after these additions.

## 4. Epistemic-uncertainty decomposition over heterogeneous vocabularies lacks theoretical justification (6Wre, T7cB, 8H82)

We clarified that the decomposition never touches raw per-model token vocabularies — it operates over a shared answer space (fixed choices A/B/C/... for MCQ; an n=5-sample answer-frequency histogram for open-ended, the same estimator LLM-TOPLA uses), so cross-model vocabulary mismatch is a non-issue by construction. We grounded the identity in established theory: the Krogh-Vedelsby ambiguity decomposition, BALD, and Lakshminarayanan et al.'s framing of deep ensembles as posterior samples.

We didn't rest this on theory alone: the rightmost plot in Figure 4 already shows a clear separation in epistemic uncertainty, and V3Fusion-Rectify's statistically significant gains are downstream of that calculation. We still agree a dedicated calibration/reliability curve (predicted uncertainty vs. empirical error) would tighten this further, and we commit to adding one.

In the follow-up round we scoped the open-ended branch more precisely: the frequency-histogram estimator is on solid ground for short-answer open-ended formats (number/word/phrase, e.g. OCR-VQA), and we pointed to span-level logit-comparison ensembling as the concrete next step for longer free-form generation.

## 5. Ablations fail to disentangle the trained fusion head from diversity-based pruning (T7cB)

We ran three experiments to isolate pruning's contribution from the head's:

1. Same compute, zero diversity (self-consistency, k=6) scores worse than a single model — so the trained/aggregating step alone doesn't explain our gains.
2. Holding the identical MLP fixed and only swapping which models get selected moves A-OKVQA accuracy by 1.6 points (85.33% to 86.94%) — isolating the selection criterion's effect.
3. We showed the diversity metric isn't just a stand-in for "pick the stronger model": CKA-diverse pairs have a smaller quality gap than average.

We're upfront that a fully controlled, fixed-head-vs-random-ensemble experiment across every dataset is still missing; we've flagged it for camera-ready.

---

## Summary of summary

We answered every meta-review point with real evidence, not just words. We showed what Focal-CKA actually contributes and cleared up the code confusion. We ran many more seeds (10 at first, then 20 more on two different GPUs) and fixed our gain formula. We added six missing baselines and fixed a broken one. We backed our uncertainty method with both theory and a working example. And we ran three experiments showing pruning helps on its own, not just the trained model. The new 40-seed runs also confirm our headline numbers and now beat PairRanker directly, not only the single best model. What's left — an uncertainty calibration plot and a random-ensemble comparison — we've clearly scoped for camera-ready.