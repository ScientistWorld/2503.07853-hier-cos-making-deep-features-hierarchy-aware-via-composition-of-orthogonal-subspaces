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
- The current code now supports the paper's 5-level CIFAR hierarchy (`paper_l5`); the next submitted run evaluates the same CE and Hier-COS comparison on that primary setting.
