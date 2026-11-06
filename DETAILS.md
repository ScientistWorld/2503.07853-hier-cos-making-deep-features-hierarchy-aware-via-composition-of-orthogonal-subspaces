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

- Public CIFAR-100 coarse/fine hierarchy, compact CNN full-split run: Hier-COS improved HOPS from 0.432 to 0.624 versus CE, with 58.10% versus 59.80% top-1 accuracy.
- Public CIFAR-100 coarse/fine hierarchy, shared WRN-16-4 full-split run: Hier-COS improved HOPS from 0.445 to 0.648 and HOPS@5 from 0.171 to 0.531, with 59.98% versus 61.21% top-1 accuracy.
- Paper 5-level CIFAR hierarchy (`paper_l5`), matched 48-epoch WRN-16-4 run: Hier-COS improved HOPS from 0.582 to 0.894, HOPS@20 from 0.222 to 0.761, AHD@5 from 2.104 to 1.257, and MS from 2.480 to 2.279 versus CE. Top-1 accuracy and FPA remain lower in this reduced single-seed reproduction: 67.65%/65.96% for Hier-COS versus 70.77%/70.29% for CE.

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

At inference, each leaf class is scored by the norm of the feature vector projected onto that leaf's subspace. The same projection-norm rule can be applied at coarser levels, producing a hierarchy-aware ranked prediction and a multi-level prediction path.

## Main Claims

The primary claim is that hierarchy-aware subspace composition improves hierarchical ranking quality and consistency without sacrificing fine-grained accuracy. Secondary claims are that the same classifier supports both fine-grained and multi-level classification, and that the HOPS metric better reflects ranked hierarchical preference than MS or AHD alone.

---

# Evaluation

## Metrics

The evaluator reads saved leaf-class score arrays and labels from `outputs/predictions/*.npz`.

Accuracy and FPA are percentages where higher is better. FPA requires both the fine prediction and its associated coarse prediction to be correct. Mistake Severity is the average LCA distance over misclassified samples, where lower is better. AHD@k averages the LCA distance of the top-k predicted leaves over all samples, where lower is better. HOPS, HOPS@5, and HOPS@20 compare the predicted ranking against the taxonomy-induced preference order; higher is better.

## Baselines and Targets

The reference target is the paper's CIFAR-100 result for Cross Entropy and Hier-COS in Table 2, plus the CIFAR-100 FPA value from Table 4. The primary runnable benchmark uses the 5-level CIFAR hierarchy table released in the public Hier-COS repository.

## Evaluation Protocol

Run training scripts to write one `.npz` prediction artifact per method, then run `bash scripts/evaluate.sh outputs/predictions`. The evaluator is method-agnostic and does not import from `method/`. For fair comparison, evaluate on the same CIFAR-100 test split and use the `paper_l5` hierarchy unless explicitly running an ablation.

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


---

# Reproduction Milestones

**Current: core_claim**

## Stop Justification
- Completed at milestone `core_claim`.
