# Hardening report (iteration 11)

Gym: CIFAR-100 hierarchical classification (Hier-COS, arXiv 2503.07853), primary metric HOPS.

All "after" numbers below come from the real evaluators run with no arguments (`bash scripts/evaluate_test.sh`, `bash scripts/evaluate_train.sh`) over each case's artifacts. Full metric vectors for both sides are in `redteam/hardening/summary.json` and `redteam/hardening/metrics.csv`; each case has its own folder under `redteam/hardening/cases/`. "Aggregate" is an illustrative linear score: the `scoring/reference.json` coefficients applied to percentages/100, HOPS as is, and distances/5. Higher is better. The external ranker's normalisation may differ, so per-metric values are always given too.

## Verdict

**Better results now score higher, beyond seed noise, and the major shortcuts are closed.**
- Before: the two most damaging shortcuts (reading the downloaded test labels, inventing a labelled population) scored HOPS 1.0 and dominated the reference on every metric.
- After: both fail.
- Two genuine improvements of the reference model beat the reference by more than the three-seed spread on the hidden test. Flip TTA gains +0.0276 aggregate and +0.0038 HOPS (+0.99 accuracy). A seed-ensemble diagnostic gains +0.036. The spreads are 0.020 aggregate and 0.0028 HOPS.
- Every weaker variant scores lower.

## What changed

| Problem (REDTEAM.md) | Fix |
|---|---|
| Test labels downloadable (`fine_labels` in the CIFAR test pickle) | `scripts/download.sh` now runs `data/prepare_cifar100.py` (stdlib only). It keeps the labelled training pickle, writes the 10,000 test images *without labels* (`test_images.u8`) and deletes the test pickle. The only workspace copy of the test labels is `eval/test/test_labels.json`, which the evaluator checks by SHA-256 |
| Ground truth, split and population taken from the submission | Evaluators own labels, row order and size: 10,000 test rows and 5,000 val rows. Submitted labels, level labels, names, flags and numbers are ignored. Wrong shapes fail |
| FPA trusted submitted path labels and only the submitted levels | FPA is recomputed from the evaluator's labels and taxonomy: top-1 leaf correct *and* all four level argmaxes correct. All four `*_level_scores_{0..3}` arrays are required; the leaf level always comes from `*_leaf_scores` |
| Taxonomy read from the mutable `data/cifar100_hierarchy.py` | Each evaluator embeds its own copy of the 5-level table |
| Baseline row taken from replaceable files on disk | CE raw outputs are frozen in `eval/train/fixed/` and `eval/test/fixed/` and re-scored by the evaluator. `cross_entropy` is a reserved key, so submitted files with that name are ignored |
| NaN/inf accepted; empty file crashed the evaluator | Non-finite, non-real, missing or misshaped arrays (and unreadable/pickled archives) score as **failed**: worst value on every metric, `_meta.methods[m].valid=false` with a reason. The evaluator still exits 0 and scores the other methods |
| Perfect-label submissions | Fine accuracy ≥ 97%, beyond the CIFAR-100 label-noise ceiling, scores as failed |
| `macro_accuracy` (exploitable by class subsetting; equals accuracy on balanced splits) | Removed. Both splits are exactly class-balanced, so it duplicated accuracy |
| `fine_coarse_consistency` covered one level only | Replaced by `path_consistency` over all four non-leaf levels, the paper's hierarchical-consistency property |
| Primary gain of a genuine improvement below seed noise; 5K hidden half-test; visible labels for half the test set | Paper protocol (Table 5, official code): train on 45K, use a fixed 5K validation split of the *training* set for the visible evaluator and best-epoch selection, and score the hidden test on all 10K test images |
| Test scoring restricted to methods present in `scores_train.json`; method names hardcoded | `evaluate_test.sh` runs with no arguments and scores every `outputs/predictions/*.npz`, keyed by file stem. No name map. `eval/train` and `eval/test` share no code and use the same artifact interface (`val_*` vs `test_*` arrays) |

The reference and baselines ran on changed code and data (45K split, unlabelled test loader, new artifact format), so they were **retrained from the final code**.

Round 1 kept the last epoch. Under the 45K split, Hier-COS then trailed a single CE seed on accuracy (77.03 vs 77.64) and FPA (76.91 vs 77.07); see `redteam/hardening/round1_last_epoch_*`. The official pipeline (`experiments/cifar-100/hier-cos.sh` → `main.py`, `val_freq=1`) tests `best.checkpoint.pth.tar`, the best validation top-1 checkpoint, for every method. Round 2 adds that missing step for both Hier-COS and CE (`--select best_val`, the default); the training method itself is unchanged. Seed 7 was declared as the reference and frozen-baseline seed before round 2 ran.

Weights: `accuracy` 0.45 → 0.50. The old setup counted recognition at 0.70 through the duplicate `macro_accuracy`, which over-weighted a constraint the paper only claims to *maintain*. Accuracy stays the heaviest constraint, below the HOPS family. `path_consistency` takes over the 0.2 of `fine_coarse_consistency`. All other coefficients and the paper numbers in `reference.json` are unchanged.

## Reference score, before and after

| | Test set | Training data / checkpoint | HOPS | HOPS@5 | HOPS@20 | Acc | Top-5 | FPA | AHD@5 | AHD@20 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Before (REDTEAM) | 5K half of test | 50K / last epoch | 0.9374 | 0.7786 | 0.8628 | 78.68 | 88.40 | 78.52 | 1.090 | 2.172 |
| Same old model, new evaluator | 10K test | 50K / last epoch | 0.9404 | 0.7806 | 0.8668 | 78.85 | 88.97 | 78.73 | 1.072 | 2.165 |
| **After (final reference, `scores_test.json`)** | 10K test | 45K / best-val (seed 7) | **0.9351** | 0.7631 | 0.8541 | 77.23 | 88.01 | 77.12 | 1.097 | 2.177 |
| Final fixed CE baseline | 10K test | 45K / best-val (seed 7) | 0.5487 | 0.0495 | 0.1634 | 77.33 | 93.85 | 76.87 | 2.274 | 3.195 |
| Paper, Hier-COS vs CE | | 45K, 5-seed mean | 0.93 vs 0.54 | 0.76 vs 0.05 | 0.85 vs 0.17 | 77.79 vs 77.77 | – | 77.37 vs 77.11 | 1.09 vs 2.25 | 2.17 vs 3.19 |

The final reproduction matches the paper's direction for the pre-declared pair:
- large gains in HOPS, HOPS@k, AHD@5/20 and MS (2.20 vs 2.39);
- accuracy on par (−0.10);
- FPA higher (+0.25, paper +0.26).

The visible split tracks the hidden test (val HOPS 0.938, accuracy 77.28).

Seeds (hidden test, reference recipe seeds 7/8/9):
- HOPS 0.9351/0.9327/0.9323 (spread 0.0028);
- accuracy 77.23/76.50/76.63 (spread 0.73);
- aggregate 2.867/2.847/2.847 (spread 0.020).

CE seeds 7/8/9:
- HOPS 0.549/0.552/0.552;
- accuracy 77.33/77.42/77.79.

The paper's HOPS effect (+0.383) is about 130× the seed spread.

## Better / weaker variants, before and after

"Before" is the old evaluator on the 5K half (REDTEAM.md), using 50K-trained models. "After, old model" rescores the same artifacts on the hardened 10K test, so only the evaluator changed. "After, retrained" uses the final 45K / best-val code, and its deltas are against the final reference (agg 2.8668).

| Variant | Before HOPS / Acc | After, old model: HOPS / Acc / Δagg vs old ref | After, retrained: HOPS / Acc / Δagg |
|---|---|---|---|
| Reference | 0.9374 / 78.68 | 0.9404 / 78.85 / 0 | 0.9351 / 77.23 / 0 |
| **Better:** flip TTA (2 views, one model) | 0.9407 / 79.38 (HOPS gain < spread) | 0.9430 / 79.74 / **+0.023** | 0.9389 / 78.22 / **+0.0276** (> seed spread 0.020) |
| **Better:** seed-ensemble diagnostic (out of bounds for scientists) | – | – | 0.9390 / 78.81 / **+0.036** |
| 10 views | 0.9379 / 78.94 | 0.9404 / 79.27 / +0.005 | – |
| 25 extra epochs | 0.9362 / 78.38 | 0.9392 / 78.71 / −0.005 | – |
| BN recalibration | 0.9363 / 78.48 | 0.9386 / 78.50 / −0.011 | – |
| **Weaker:** 1 epoch (before) / 5 epochs (after) | 0.5388 / 8.80 | 0.5410 / 8.88 / −2.37 | 0.6717 / 23.18 / −1.76 |
| **Weaker:** uniform level weights (paper Table 8) | – | – | 0.9343 / 76.12 / −0.017 |
| **Weaker:** reversed level weights (paper Table 8) | – | – | 0.8799 / 22.07 / −0.975 |
| **Weaker:** single-axis subspaces (paper Fig. 4b) | – | – | 0.6117 / 77.45 / −0.931 |
| Flat CE baseline | 0.5545 / 77.92 | 0.5556 / 78.50 / −1.28 | 0.5487 / 77.33 / −1.27 |
| Honest alternative: CE + taxonomy re-rank | 0.9305 / 77.92 | – | 0.9284 / 77.33 / −0.025 |

Before, a genuine improvement's primary gain was below seed noise, and the vector was not clearly ordered. After, both retrained "better" variants are ahead by more than the seed spread on HOPS (+0.0038, +0.0039 vs 0.0028), accuracy (+0.99, +1.58 vs 0.73) and aggregate. Every paper ablation and the undertrained run falls below. Uniform weights only falls slightly (−0.017 agg, −1.1 accuracy), matching the paper's finding that it mainly hurts accuracy. TTA is a paired comparison on the seed-7 model, so its gain does not include seed variance.

## Shortcuts: fix and score, before → after (hidden test)

The final reference is HOPS 0.9351 / aggregate 2.8668.

| Shortcut | Before | Fix | After |
|---|---|---|---|
| Read test labels from the downloaded pickle; perfect taxonomy ranking | HOPS **1.0**, dominates all metrics | No labels in `data/downloads/`; evaluator-owned labels; plausibility ceiling | Labels unavailable from the gym. Same submission with externally obtained labels: **failed** (accuracy 100% ≥ 97% ceiling; all metrics worst) |
| Constant prediction + invented labels | HOPS **1.0**, dominates | Submitted labels ignored, fixed population | HOPS 0.326, accuracy 1.0% (agg −2.90) |
| Keep only correct rows | HOPS 0.9973, accuracy 100% | Exact row count | **failed** (shape 7723 ≠ 10000) |
| Self-reported path labels | FPA 100% vs 78.5 (dominates) | FPA from evaluator truth | Exact tie with reference; labels ignored |
| Omit leaf/other path levels | FPA 87.3% vs 78.5 (dominates) | All four levels required | **failed** (missing arrays) |
| Collapse `PAPER_L5_ROWS` in `data/` + trivial path arrays | HOPS **1.0**, all AHD/MS 0 | Evaluator-owned taxonomy, width check | **failed** (width 1 ≠ 2). With the reference arrays and the collapsed data file: exact tie, no effect |
| Replace CE baseline artifact | CE row 0.5545 → 0.3056 (comparison moved) | Frozen evaluator-owned baseline, reserved key | CE row unchanged (0.5487); reserved file ignored; candidate ties |
| Prewritten scores / claimed fields in artifact | tie (spoof ignored) | Unchanged: always recomputed | tie (spoof ignored) |
| NaN / +inf leaf scores | accepted (0.259 / 0.306), FPA still 78.5 | Finite check | **failed** |
| Empty arrays | evaluator crashed (exit 1) | Shape check, no crash | **failed** (evaluator exits 0) |
| Zero leaf scores | 0.306, FPA still 78.5 | FPA needs the leaf | 0.306, accuracy 1.0%, FPA 0.99 |
| Rename CE artifact to `hier_cos.npz` | CE values | – | CE values (0.5487) |

## Shortcuts and gaps left (stated plainly)

- **Externally obtained test labels used partially.** CIFAR-100 test labels are public, and this machine's shared dataset cache also holds the labelled test pickle. Using them on 15% of rows stays under the 97% ceiling and scores HOPS 0.946, accuracy 80.8%, **aggregate +0.092**, more than any honest variant. No evaluator can tell this from a better classifier. It is forbidden in `scoring/CONSTRAINTS.md` and left to the compliance grader.
- **Training on the 5K validation split, or on test images transductively.** These cannot be detected from outputs (the hidden test sees only test rows). Forbidden in CONSTRAINTS; left to the grader.
- **Ensembles / bigger or pretrained backbones.** Genuinely better predictions (the ensemble diagnostic scores +0.036). This is a rules matter (CONSTRAINTS: single model, comparable scale), not a scoring loophole.
- **Leaf-derived level predictions** (ancestors of the predicted leaf). This honest inference rule makes `path_consistency` 100% and FPA = accuracy, for +0.001 aggregate, a tie. `path_consistency` therefore has little headroom (reference 99.58%) and mainly penalises inconsistent multi-level outputs (CE 97.3%). Left as is.
- **Taxonomy re-ranking of CE.** This honest alternative reaches HOPS 0.928, close to the reference. The test separates it (agg −0.025, top-5 85.8 vs 88.0), but only by about the seed spread on aggregate. That is a headroom limitation of HOPS, not a loophole.
- **Accuracy on seed means.** The pre-declared seed-7 pair matches the paper (77.23 vs 77.33 accuracy, FPA +0.25). On three-seed means, Hier-COS accuracy is 0.72 points below CE (76.79 vs 77.51) and FPA 0.31 below (76.61 vs 76.93). The paper reports parity (77.79 vs 77.77), so the "no loss in accuracy" constraint is reproduced only to within about one point.
- **The reference seed sits high.** Seed 7 is the best of the three reference seeds on the hidden test (aggregate 2.867 vs 2.847/2.847). A rerun of the reference recipe with another seed should be expected to score about 0.02 aggregate (about 0.003 HOPS) lower.

## Headroom check

Final reference on the hidden test, against each metric's bound:
- HOPS 0.935 (bound 1); HOPS@5 0.763 (1); HOPS@20 0.854 (1)
- accuracy 77.2% (100); top-5 88.0% (100); coarse accuracy 86.9% (100); FPA 77.1% (100)
- MS 2.20 (≥ 1 for any error in this tree); AHD@1 0.50 (0); AHD@5 1.097 (perfect-ranking value 0.80); AHD@20 2.177 (2.0625)
- path_consistency 99.58% (100)

**No scored metric is at its bound.** `path_consistency` is near it, and is a constraint. Three facts show the room is reachable:
- single-model TTA and the ensemble score higher;
- the 50K-trained models from earlier iterations reach HOPS 0.940 and accuracy 78.9 on the same test;
- the reference's top-5 accuracy is about 6 points below CE.

Because the reference is not at the bound of any metric, no extra experiment was added.

## Files

- Evaluators: `eval/test/evaluate.py` (hidden test: labels, taxonomy, frozen baseline), `eval/train/evaluate.py` (visible val). Artifact helper: `data/prediction_format.py`.
- Data: `scripts/download.sh`, `data/prepare_cifar100.py`, `data/cifar100_val_split.json`, `data/cifar100_dataset.py`.
- Method / baseline: `--select best_val` (default), paper-ablation flags `--level-weights`, `--subspace`, inference flag `--tta`. Baseline freezing: `baseline/freeze_fixed_baseline.py` (via `FREEZE_BASELINE=1 scripts/reproduce.sh`).
- Hardening harness: `redteam/hardening/{run_hardening.sh,build_cases.py,score_cases.py,ensemble_predict.py}`. Round 1 (last-epoch) record: `redteam/hardening/round1_last_epoch_*`.
- `scripts/run.sh` reproduces the reference end to end: baseline + Hier-COS seed 7, freeze, val and test scoring.
- Compute: two 2-GPU jobs, about 1 h 50 min and 2 h 10 min.
