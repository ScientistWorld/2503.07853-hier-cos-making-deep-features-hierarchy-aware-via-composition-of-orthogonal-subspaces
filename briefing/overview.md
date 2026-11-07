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
