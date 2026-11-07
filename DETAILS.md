# Overview

- **Paper ID:** 2503.07853
- **Title:** Hier-COS: Making Deep Features Hierarchy-aware via Composition of Orthogonal Subspaces
- **Domain:** Computer Vision / Hierarchical Classification
- **TL;DR:** The paper maps image features into taxonomy-composed orthogonal subspaces so predictions are accurate, hierarchy-aware, and consistent across levels.

## Short Summary

The paper argues that common hierarchical metrics miss whether top-k predictions are ordered according to the class taxonomy. It proposes Hier-COS, where every taxonomy node owns an orthogonal coordinate and every class is classified by projection onto a composed subspace. It also introduces HOPS, a rank-based hierarchical score. On CIFAR-100, the paper reports Hier-COS improves HOPS from 0.54 for Cross Entropy to 0.93 while maintaining similar top-1 accuracy.

## Key Results

- CIFAR-100 Table 2 reports Hier-COS at 77.79% accuracy and 0.93 HOPS.
- CIFAR-100 Table 4 reports Hier-COS FPA of 77.37%, above the listed hierarchical baselines.
- iNaturalist-19 with frozen ViT features reports a large HOPS gain, 0.53 to 0.98, and higher top-1 accuracy.

## Current Reproduction

Benchmark protocol (paper Table 5 and the official pipeline): WRN-28-8, 100 epochs, SGD with cosine decay. Training uses 45K images, with a fixed 5K validation split for best-epoch selection and development scoring. The final scores come from the 10,000 official test images, whose labels are hidden. The taxonomy is the paper's 5-level CIFAR hierarchy.

- Reference (Hier-COS, seed 7) versus the flat cross-entropy baseline (seed 7), on the hidden test: HOPS 0.935 vs 0.549 (paper 0.93 vs 0.54). HOPS@5 is 0.763 vs 0.050 and HOPS@20 0.854 vs 0.163. AHD@5 is 1.097 vs 2.274, AHD@20 2.177 vs 3.195, and MS 2.20 vs 2.39. Accuracy is 77.23% vs 77.33% (paper 77.79 vs 77.77), and full-path accuracy 77.12% vs 76.87% (paper 77.37 vs 77.11).
- Over three seeds the HOPS gain (0.933 vs 0.551) is roughly 130 times the seed spread (0.003). On mean top-1 accuracy, Hier-COS is 0.7 points lower (76.79 vs 77.51), and its top-5 accuracy is much lower (87.8% vs 93.9%).
- Visible validation scores track the hidden test: reference HOPS is 0.938 on val and 0.935 on test.

---

# Problem Definition

## Research Question

How can an image classifier use a known class taxonomy so that its ranked predictions respect semantic similarity while still retaining fine-grained top-1 accuracy?

## Why It Matters

Flat classifiers treat every wrong label as equally bad. In hierarchical domains such as species, products, or object taxonomies, confusing two nearby classes is preferable to confusing distant classes, and users may also need consistent predictions at coarse and fine levels.

## Success Criteria

A successful system should classify fine-grained labels accurately, make less severe mistakes when it is wrong, rank semantically closer alternatives above distant ones, and produce coarse-level predictions that agree with the fine-level prediction path.

---

# Paper's Method

## Key Contribution

Hier-COS represents a hierarchy as a vector space with one fixed orthonormal basis vector per taxonomy node. Each class is assigned the subspace spanned by its ancestors, itself, and its descendants, so related classes share dimensions and distant classes share fewer or none.

## Approach

A visual backbone maps an image to features, and a transformation module maps those features into the hierarchy-node coordinate space. Training uses a weighted path distribution target: the true nodes along the sample's taxonomy path receive increasing weights toward the leaf, and all other nodes receive zero mass. The loss combines KL divergence between this path distribution and a regularizer that encourages one active coordinate per true hierarchy level.

Training follows the paper's CIFAR-100 recipe: WRN-28-8 with the transformation head, 100 epochs, batch 64, SGD with cosine decay, alpha 0.05, on the 45K training split. As in the official pipeline, the checkpoint with the best validation top-1 accuracy is kept.

At inference, each leaf class is scored by the norm of the feature vector projected onto that leaf's subspace. The same projection-norm rule can be applied at coarser levels, producing a hierarchy-aware ranked prediction and a multi-level prediction path.

## Main Claims

The primary claim is that hierarchy-aware subspace composition improves hierarchical ranking quality and consistency without sacrificing fine-grained accuracy. Secondary claims are that the same classifier supports both fine-grained and multi-level classification, and that the HOPS metric better reflects ranked hierarchical preference than MS or AHD alone.

---

# Evaluation

## Data protocol

CIFAR-100 with the paper's 5-level class taxonomy (`paper_l5` in `data/cifar100_hierarchy.py`; 2, 4, 8 and 20 nodes on the four non-leaf levels, 100 leaves).

| Split | Images | Labels | Use |
|---|---|---|---|
| train | 45,000 (official training set minus the validation split) | visible | training |
| val | 5,000 fixed images of the official training set, 50 per class (`data/cifar100_val_split.json`) | visible | development, scored by `scripts/evaluate.sh` |
| test | the 10,000 official test images (`data/downloads/cifar-100-python/test_images.u8`) | withheld | hidden final scoring |

`scripts/download.sh` writes the training pickle and the unlabelled test images. Both scored splits are exactly class-balanced.

## Prediction artifact

One file per method: `outputs/predictions/<method_key>.npz`. The file stem is the method's name in every score file; `cross_entropy` is reserved for the fixed baseline. Required arrays, all real-valued and finite, higher = more preferred:

| Array | Shape | Rows |
|---|---|---|
| `val_leaf_scores` | (5000, 100) | in `val_indices` order |
| `val_level_scores_{l}`, l = 0..3 | (5000, K_l), K = (2, 4, 8, 20) | same |
| `test_leaf_scores` | (10000, 100) | official test order |
| `test_level_scores_{l}`, l = 0..3 | (10000, K_l) | same |

Column j of `*_level_scores_{l}` is node j of level l (level 0 = children of the root, level 3 = parents of the leaves), i.e. value j in column `4 - l` of `PAPER_L5_ROWS`. `data/prediction_format.py` writes this format. Labels, paths, names, flags or numbers stored in the artifact are ignored: the evaluators own the ground truth, the taxonomy, the baseline and the split sizes.

A submission is scored as **failed** (worst value on every metric) if an array is missing, has the wrong shape or a non-real dtype, or contains NaN/inf. It also fails if its fine accuracy is at least 97%, which is beyond the CIFAR-100 label-noise ceiling and indicates label lookup.

## Metrics

Computed per split from the raw scores. The leaf ranking is the descending order of `*_leaf_scores`, and a level's prediction is the argmax of its `*_level_scores_{l}`. LCA distance between two leaves is the number of levels above the leaves up to their lowest common ancestor (0 to 5).

| Metric | Direction | Definition |
|---|---|---|
| `hops` (primary) | higher | Hierarchically Ordered Preference Score: weighted disagreement between the predicted ranking of all 100 classes and the taxonomy-induced preference order (true class, then classes by increasing LCA distance), normalised by the worst order. Weights halve at each new preference rank and decay linearly within a rank |
| `hops@5`, `hops@20` | higher | the same, counting only the top 5 / 20 ranked positions |
| `accuracy`, `top5_accuracy` | higher | fine-label top-1 / top-5 accuracy (%) |
| `coarse_accuracy` | higher | accuracy (%) at level 3 (20 nodes) |
| `fpa` | higher | full-path accuracy (%): top-1 leaf and all four level predictions correct |
| `path_consistency` | higher | % of examples whose four level predictions are exactly the ancestors of the top-1 leaf |
| `ms` | lower | mean LCA distance of top-1 mistakes (misclassified examples only) |
| `ahd@1`, `ahd@5`, `ahd@20` | lower | mean LCA distance of the top-1 / top-5 / top-20 ranked classes over all examples |

## Baselines and targets

Each score file has a `cross_entropy` row, recomputed by the evaluator from its own frozen copy of a flat cross-entropy classifier's outputs. That classifier is the same backbone, recipe and training split as the reference, with superclass scores formed as the sum of leaf probabilities. `scoring/reference.json` lists the paper's CIFAR-100 numbers (Table 2 and the CIFAR-100 part of Table 4) for this baseline and for the paper's best hierarchy-aware system.

## Protocol

Write the artifact, then run `bash scripts/evaluate.sh` (writes `scoring/scores.json` for the val split). `scripts/evaluate_train.sh` is the same val scoring written to `scoring/scores_train.json`. The hidden test (`scripts/evaluate_test.sh`, no arguments) applies identical code and rules to the `test_*` arrays of every artifact in `outputs/predictions/`. It never retrains or runs a model.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 19m | **GPU**: 0.1h
- **Jobs**: 4 total (3 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-13 20:31] - none
- Read the paper and extracted the Hier-COS algorithm and HOPS metric.
- Implemented the initial redistributable CIFAR-100 reproduction scaffold.
- Prepared a first compute job to test whether the method runs end-to-end.

### [2026-06-13 20:46] - method_runs
- Completed a GPU smoke job on 6,000 CIFAR-100 training examples and 2,000 test examples.
- Verified that both the Cross Entropy baseline and the Hier-COS implementation train, write prediction artifacts, and score through `scripts/evaluate.sh`.
- Smoke result does not yet support the core claim: Hier-COS improved HOPS@5 over the baseline, but accuracy and several hierarchy metrics remained weak.

### [2026-06-13 20:52] - method_runs
- Full-split job `56c4be57-6dc` failed before training due to a missing `--weight-decay` CLI argument in the trainer scripts.
- Fixed the argument mismatch and prepared a rerun of the same full-split experiment.

### [2026-06-13 21:00] - core_claim
- Completed a full-split CIFAR-100 run with matched CE and Hier-COS models.
- Hier-COS improved the primary hierarchical ranking metric HOPS from 0.432 to 0.624 and improved HOPS@5 from 0.163 to 0.504.
- Top-1 accuracy remained comparable but slightly lower than CE, 58.10% versus 59.80%, so the next run upgrades the shared backbone toward the paper's WideResNet setting.

### [2026-06-13 21:08] - core_claim_plus
- Completed a second full-split CIFAR-100 setting with a shared WRN-16-4 backbone.
- Hier-COS improved HOPS from 0.445 to 0.648, HOPS@5 from 0.171 to 0.531, HOPS@20 from 0.565 to 0.707, and AHD@5 from 1.385 to 1.100.
- Accuracy remained comparable but lower than CE, 59.98% versus 61.21%, so the gym records both the hierarchy benefit and the fine-grained accuracy constraint.

</details>

### Iteration 2: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 16m | **GPU**: 0.3h
- **Jobs**: 9 total (8 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-13 20:43] - method_runs
- Implemented Hier-COS with fixed orthonormal node-coordinate bases, composed ancestor/self/descendant subspaces, weighted path KL loss, and projection-norm inference.
- Completed a smoke run on CIFAR-100 subsets and evaluated saved prediction artifacts through the method-independent evaluator.

### [2026-06-13 20:52] - core_claim
- Completed a full-split public CIFAR-100 run with the compact CNN backbone.
- Hier-COS improved hierarchical ranking quality over cross entropy: HOPS 0.6237 vs 0.4319 and HOPS@5 0.5043 vs 0.1631, while tracking accuracy and FPA constraints.

### [2026-06-13 20:59] - core_claim_plus
- Completed a full-split public CIFAR-100 run with the stronger WRN-16-4 backbone.
- Hier-COS again improved the core hierarchy-aware metrics: HOPS 0.6482 vs 0.4445, HOPS@5 0.5314 vs 0.1708, HOPS@20 0.7066 vs 0.5655, and AHD@5 1.0998 vs 1.3855.
- The workspace is packaged with download, method, baseline, evaluation, and reproduction scripts; `python validate.py --compare` passes.

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 26m | **GPU**: 0.8h
- **Jobs**: 10 total (9 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-13 22:54] - secondary_claims
- Completed the official-style single-seed CIFAR-100 WRN-28-8 run on the paper's 5-level hierarchy.
- Reproduced the paper's primary hierarchy-aware ranking effect: HOPS 0.940 for Hier-COS versus 0.556 for CE, close to the reported 0.930 versus 0.540.
- Reproduced the key constraint side: Hier-COS kept comparable or better accuracy/FPA, reaching 78.85% accuracy and 78.73 FPA versus CE at 78.50% and 78.04.

### [2026-06-13 22:03] - core_claim_plus
- Audited the continued workspace and restored confidence in the previous `core_claim_plus` milestone.
- Verified the implementation contains the paper's actual Hier-COS computation and that evaluation reads method outputs independently.
- Cleaned tracked generated artifacts and decoupled the CE baseline from the method module before submitting the higher-fidelity WRN-28-8 follow-up.

### [2026-06-13 21:52] - core_claim_plus
- Completed matched WRN-16-4 runs on the paper's 5-level CIFAR hierarchy.
- Hier-COS reproduced the central hierarchy-aware ranking effect over CE while tracking accuracy and full-path accuracy constraints.
- Added an official-style WRN-28-8/SGD CIFAR configuration and queued a paper-scale single-seed follow-up.

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 10 total (9 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-13 22:54] - secondary_claims
- Completed the official-style single-seed CIFAR-100 WRN-28-8 run on the paper's 5-level hierarchy.
- Reproduced the paper's primary hierarchy-aware ranking effect: HOPS 0.940 for Hier-COS versus 0.556 for CE, close to the reported 0.930 versus 0.540.
- Reproduced the key constraint side: Hier-COS kept comparable or better accuracy/FPA, reaching 78.85% accuracy and 78.73 FPA versus CE at 78.50% and 78.04.

### [2026-06-13 22:03] - core_claim_plus
- Audited the continued workspace and restored confidence in the previous `core_claim_plus` milestone.
- Verified the implementation contains the paper's actual Hier-COS computation and that evaluation reads method outputs independently.
- Cleaned tracked generated artifacts and decoupled the CE baseline from the method module before submitting the higher-fidelity WRN-28-8 follow-up.

### [2026-06-13 21:52] - core_claim_plus
- Completed matched WRN-16-4 runs on the paper's 5-level CIFAR hierarchy.
- Hier-COS reproduced the central hierarchy-aware ranking effect over CE while tracking accuracy and full-path accuracy constraints.
- Added an official-style WRN-28-8/SGD CIFAR configuration and queued a paper-scale single-seed follow-up.

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 10 total (9 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-13 22:54] - secondary_claims
- Completed the official-style single-seed CIFAR-100 WRN-28-8 run on the paper's 5-level hierarchy.
- Reproduced the paper's primary hierarchy-aware ranking effect: HOPS 0.940 for Hier-COS versus 0.556 for CE, close to the reported 0.930 versus 0.540.
- Reproduced the key constraint side: Hier-COS kept comparable or better accuracy/FPA, reaching 78.85% accuracy and 78.73 FPA versus CE at 78.50% and 78.04.

### [2026-06-13 22:03] - core_claim_plus
- Audited the continued workspace and restored confidence in the previous `core_claim_plus` milestone.
- Verified the implementation contains the paper's actual Hier-COS computation and that evaluation reads method outputs independently.
- Cleaned tracked generated artifacts and decoupled the CE baseline from the method module before submitting the higher-fidelity WRN-28-8 follow-up.

### [2026-06-13 21:52] - core_claim_plus
- Completed matched WRN-16-4 runs on the paper's 5-level CIFAR hierarchy.
- Hier-COS reproduced the central hierarchy-aware ranking effect over CE while tracking accuracy and full-path accuracy constraints.
- Added an official-style WRN-28-8/SGD CIFAR configuration and queued a paper-scale single-seed follow-up.

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
- **Jobs**: 12 total (11 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-13 22:54] - secondary_claims
- Completed the official-style single-seed CIFAR-100 WRN-28-8 run on the paper's 5-level hierarchy.
- Reproduced the paper's primary hierarchy-aware ranking effect: HOPS 0.940 for Hier-COS versus 0.556 for CE, close to the reported 0.930 versus 0.540.
- Reproduced the key constraint side: Hier-COS kept comparable or better accuracy/FPA, reaching 78.85% accuracy and 78.73 FPA versus CE at 78.50% and 78.04.

### [2026-06-13 22:03] - core_claim_plus
- Audited the continued workspace and restored confidence in the previous `core_claim_plus` milestone.
- Verified the implementation contains the paper's actual Hier-COS computation and that evaluation reads method outputs independently.
- Cleaned tracked generated artifacts and decoupled the CE baseline from the method module before submitting the higher-fidelity WRN-28-8 follow-up.

### [2026-06-13 21:52] - core_claim_plus
- Completed matched WRN-16-4 runs on the paper's 5-level CIFAR hierarchy.
- Hier-COS reproduced the central hierarchy-aware ranking effect over CE while tracking accuracy and full-path accuracy constraints.
- Added an official-style WRN-28-8/SGD CIFAR configuration and queued a paper-scale single-seed follow-up.

</details>

### Iteration 7: gpt-5.5 (packager)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 2m | **GPU**: 0.0h
- **Jobs**: 12 total (11 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-13 22:54] - secondary_claims
- Completed the official-style single-seed CIFAR-100 WRN-28-8 run on the paper's 5-level hierarchy.
- Reproduced the paper's primary hierarchy-aware ranking effect: HOPS 0.940 for Hier-COS versus 0.556 for CE, close to the reported 0.930 versus 0.540.
- Reproduced the key constraint side: Hier-COS kept comparable or better accuracy/FPA, reaching 78.85% accuracy and 78.73 FPA versus CE at 78.50% and 78.04.

### [2026-06-13 22:03] - core_claim_plus
- Audited the continued workspace and restored confidence in the previous `core_claim_plus` milestone.
- Verified the implementation contains the paper's actual Hier-COS computation and that evaluation reads method outputs independently.
- Cleaned tracked generated artifacts and decoupled the CE baseline from the method module before submitting the higher-fidelity WRN-28-8 follow-up.

### [2026-06-13 21:52] - core_claim_plus
- Completed matched WRN-16-4 runs on the paper's 5-level CIFAR hierarchy.
- Hier-COS reproduced the central hierarchy-aware ranking effect over CE while tracking accuracy and full-path accuracy constraints.
- Added an official-style WRN-28-8/SGD CIFAR configuration and queued a paper-scale single-seed follow-up.

</details>

### Iteration 8: gpt-5.5 (verifier)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 7m | **GPU**: 0.0h
- **Jobs**: 12 total (11 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-13 23:21] - secondary_claims
- Completed final verification for gym readiness.
- Fixed method/problem separation leakage in the CIFAR hierarchy data utilities and method-agnostic documentation.
- Aligned top-level `scores.json` and `eval/metrics.py` with every nonzero metric coefficient in `reference.json`.

### [2026-06-13 22:54] - secondary_claims
- Completed the official-style single-seed CIFAR-100 WRN-28-8 run on the paper's 5-level hierarchy.
- Reproduced the paper's primary hierarchy-aware ranking effect: HOPS 0.940 for Hier-COS versus 0.556 for CE, close to the reported 0.930 versus 0.540.
- Reproduced the key constraint side: Hier-COS kept comparable or better accuracy/FPA, reaching 78.85% accuracy and 78.73 FPA versus CE at 78.50% and 78.04.

### [2026-06-13 22:03] - core_claim_plus
- Audited the continued workspace and restored confidence in the previous `core_claim_plus` milestone.
- Verified the implementation contains the paper's actual Hier-COS computation and that evaluation reads method outputs independently.
- Cleaned tracked generated artifacts and decoupled the CE baseline from the method module before submitting the higher-fidelity WRN-28-8 follow-up.

### [2026-06-13 21:52] - core_claim_plus
- Completed matched WRN-16-4 runs on the paper's 5-level CIFAR hierarchy.
- Hier-COS reproduced the central hierarchy-aware ranking effect over CE while tracking accuracy and full-path accuracy constraints.
- Added an official-style WRN-28-8/SGD CIFAR configuration and queued a paper-scale single-seed follow-up.

</details>

### Iteration 9: gpt-5.5 (verifier)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 7m | **GPU**: 0.0h
- **Jobs**: 12 total (11 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-13 23:21] - secondary_claims
- Completed final verification for gym readiness.
- Fixed method/problem separation leakage in the CIFAR hierarchy data utilities and method-agnostic documentation.
- Aligned top-level `scores.json` and `eval/metrics.py` with every nonzero metric coefficient in `reference.json`.

### [2026-06-13 22:54] - secondary_claims
- Completed the official-style single-seed CIFAR-100 WRN-28-8 run on the paper's 5-level hierarchy.
- Reproduced the paper's primary hierarchy-aware ranking effect: HOPS 0.940 for Hier-COS versus 0.556 for CE, close to the reported 0.930 versus 0.540.
- Reproduced the key constraint side: Hier-COS kept comparable or better accuracy/FPA, reaching 78.85% accuracy and 78.73 FPA versus CE at 78.50% and 78.04.

### [2026-06-13 22:03] - core_claim_plus
- Audited the continued workspace and restored confidence in the previous `core_claim_plus` milestone.
- Verified the implementation contains the paper's actual Hier-COS computation and that evaluation reads method outputs independently.
- Cleaned tracked generated artifacts and decoupled the CE baseline from the method module before submitting the higher-fidelity WRN-28-8 follow-up.

### [2026-06-13 21:52] - core_claim_plus
- Completed matched WRN-16-4 runs on the paper's 5-level CIFAR hierarchy.
- Hier-COS reproduced the central hierarchy-aware ranking effect over CE while tracking accuracy and full-path accuracy constraints.
- Added an official-style WRN-28-8/SGD CIFAR configuration and queued a paper-scale single-seed follow-up.

</details>

### Iteration 10: gpt-6.1-sol (redteam)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 13m | **GPU**: 2.5h
- **Jobs**: 12 total (11 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-13 23:21] - secondary_claims
- Completed final verification for gym readiness.
- Fixed method/problem separation leakage in the CIFAR hierarchy data utilities and method-agnostic documentation.
- Aligned top-level `scores.json` and `eval/metrics.py` with every nonzero metric coefficient in `reference.json`.

### [2026-06-13 22:54] - secondary_claims
- Completed the official-style single-seed CIFAR-100 WRN-28-8 run on the paper's 5-level hierarchy.
- Reproduced the paper's primary hierarchy-aware ranking effect: HOPS 0.940 for Hier-COS versus 0.556 for CE, close to the reported 0.930 versus 0.540.
- Reproduced the key constraint side: Hier-COS kept comparable or better accuracy/FPA, reaching 78.85% accuracy and 78.73 FPA versus CE at 78.50% and 78.04.

### [2026-06-13 22:03] - core_claim_plus
- Audited the continued workspace and restored confidence in the previous `core_claim_plus` milestone.
- Verified the implementation contains the paper's actual Hier-COS computation and that evaluation reads method outputs independently.
- Cleaned tracked generated artifacts and decoupled the CE baseline from the method module before submitting the higher-fidelity WRN-28-8 follow-up.

### [2026-06-13 21:52] - core_claim_plus
- Completed matched WRN-16-4 runs on the paper's 5-level CIFAR hierarchy.
- Hier-COS reproduced the central hierarchy-aware ranking effect over CE while tracking accuracy and full-path accuracy constraints.
- Added an official-style WRN-28-8/SGD CIFAR configuration and queued a paper-scale single-seed follow-up.

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 26m | **GPU**: 6.4h
- **Jobs**: 14 total (13 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-11-07] - secondary_claims (hardening)
- Hardened the hidden test. Test labels are removed from scientist data; labels, taxonomy and the CE baseline are evaluator-owned; the evaluators check artifact shapes and finiteness and fail implausible runs; the full 10K test set is scored. The paper's 45K/5K split is used with best-validation checkpoint selection.
- Retrained the reference (Hier-COS seed 7) and the CE baseline from the final code. On the hidden test: HOPS 0.935 vs 0.549, accuracy 77.23 vs 77.33, FPA 77.12 vs 76.87.
- All major red-team shortcuts now fail or tie. Genuine improvements score above the seed spread and the paper's ablations score below. See HARDENING.md.

### [2026-06-13 23:21] - secondary_claims
- Completed final verification for gym readiness.
- Fixed method/problem separation leakage in the CIFAR hierarchy data utilities and method-agnostic documentation.
- Aligned top-level `scores.json` and `eval/metrics.py` with every nonzero metric coefficient in `reference.json`.

### [2026-06-13 22:54] - secondary_claims
- Completed the official-style single-seed CIFAR-100 WRN-28-8 run on the paper's 5-level hierarchy.
- Reproduced the paper's primary hierarchy-aware ranking effect: HOPS 0.940 for Hier-COS versus 0.556 for CE, close to the reported 0.930 versus 0.540.
- Reproduced the key constraint side: Hier-COS kept comparable or better accuracy/FPA, reaching 78.85% accuracy and 78.73 FPA versus CE at 78.50% and 78.04.

### [2026-06-13 22:03] - core_claim_plus
- Audited the continued workspace and restored confidence in the previous `core_claim_plus` milestone.
- Verified the implementation contains the paper's actual Hier-COS computation and that evaluation reads method outputs independently.
- Cleaned tracked generated artifacts and decoupled the CE baseline from the method module before submitting the higher-fidelity WRN-28-8 follow-up.

### [2026-06-13 21:52] - core_claim_plus
- Completed matched WRN-16-4 runs on the paper's 5-level CIFAR hierarchy.
- Hier-COS reproduced the central hierarchy-aware ranking effect over CE while tracking accuracy and full-path accuracy constraints.
- Added an official-style WRN-28-8/SGD CIFAR configuration and queued a paper-scale single-seed follow-up.

</details>


---

# Reproduction Milestones

**Current: secondary_claims**

## Stop Justification
- Completed at milestone `secondary_claims`.
