# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

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

## Audit Note
- The previous agent claimed `core_claim_plus` from public CIFAR hierarchy runs. This continuation reset starts at `core_claim` and submits a direct `paper_l5` run before restoring or advancing that claim.

### [2026-06-13 21:18] - core_claim
- Continuing audit reset the active milestone to `core_claim`.
- Verified the implementation is the paper's algorithm and evaluation is independent, then extended the benchmark to the paper's 5-level CIFAR hierarchy.
- Submitting a WRN-16-4 CE vs Hier-COS run on `paper_l5` before restoring or advancing the milestone.

### [2026-06-13 21:28] - core_claim
- Completed the first `paper_l5` WRN-16-4 run.
- Hier-COS improved the hierarchy-aware benefit metrics over CE on the paper hierarchy, including HOPS 0.843 vs 0.593 and AHD@5 1.495 vs 2.050.
- Accuracy and FPA lagged the CE constraint metrics, and training was still improving at epoch 16; submitting a longer method run to improve the constraint side.

### [2026-06-13 21:36] - core_claim
- Completed the 48-epoch Hier-COS run on `paper_l5`.
- Hier-COS now improves both hierarchy-aware metrics and the accuracy/FPA constraints against the existing 12-epoch CE artifact.
- Submitting a matched 48-epoch CE run before restoring `core_claim_plus`, so the comparison is fair on training duration.

### [2026-06-13 21:44] - core_claim
- Completed the matched 48-epoch CE run on `paper_l5`.
- Hier-COS still strongly improves HOPS and AHD over CE, but CE has higher top-1 accuracy and FPA under matched epochs.
- Fixed the Hier-COS loss to match the official implementation more closely and submitting a new 48-epoch method run.

### [2026-06-13 21:53] - core_claim
- The official-loss alpha `0.05` run completed but underfit the reduced setup: 37.59% accuracy with HOPS 0.788.
- Since the paper's alpha sensitivity includes smaller values with similar hierarchy performance, submitting an alpha `0.005` Hier-COS run.

### [2026-06-13 22:03] - core_claim_plus
- Completed the official-loss alpha `0.005` run on the paper's 5-level CIFAR hierarchy.
- On matched 48-epoch WRN-16-4 runs, Hier-COS improves the central hierarchy-aware metrics over CE: HOPS 0.894 vs 0.582, HOPS@20 0.761 vs 0.222, AHD@5 1.257 vs 2.104, AHD@20 2.236 vs 3.058, and MS 2.279 vs 2.480.
- Accuracy and FPA remain lower than CE in this reduced single-seed setup, but the benchmark now measures both benefit and constraint metrics on the paper's primary CIFAR hierarchy.
