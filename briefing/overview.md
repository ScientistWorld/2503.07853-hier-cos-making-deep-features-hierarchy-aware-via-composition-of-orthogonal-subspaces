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
- Paper 5-level CIFAR hierarchy (`paper_l5`), official-style WRN-28-8 100-epoch run: Hier-COS reproduced the paper-scale CIFAR claim, improving HOPS from 0.556 to 0.940 and HOPS@20 from 0.176 to 0.867 versus CE while slightly improving accuracy/FPA from 78.50/78.04 to 78.85/78.73.
