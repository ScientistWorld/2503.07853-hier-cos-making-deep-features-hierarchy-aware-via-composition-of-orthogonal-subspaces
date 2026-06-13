# Progress

## What Works

- Paper read and core algorithm identified: fixed orthonormal node bases, composed ancestor/self/descendant subspaces, weighted path KL loss, sparsity regularizer, and projection-norm inference.
- Implemented CIFAR-100 data loading, a public CIFAR hierarchy, Hier-COS training, a Cross Entropy baseline, and a method-independent evaluator.
- Added a selectable CIFAR WideResNet-style backbone so CE and Hier-COS can be compared on a backbone family closer to the paper's setup.
- Added support for the paper's 5-level CIFAR hierarchy from the public Hier-COS target table while preserving the earlier public coarse/fine hierarchy as an ablation option.

## Results

- Completed 2-epoch smoke run on 6,000 CIFAR-100 training images and 2,000 test images.

| method | accuracy | HOPS | HOPS@5 | HOPS@20 | FPA |
|---|---:|---:|---:|---:|---:|
| cross_entropy | 9.05 | 0.0497 | 0.0139 | 0.2906 | 7.25 |
| hier_cos | 6.10 | 0.0512 | 0.0325 | 0.2877 | 5.10 |

- This reaches `method_runs`: the paper's algorithm executes end-to-end and is evaluated from saved outputs.
- This does not reach `core_claim`: the current subset/epoch setting does not yet demonstrate a clear Hier-COS advantage across the paper's benefit and constraint metrics.

- Completed full-split run with the initial compact CNN backbone.

| method | accuracy | HOPS | HOPS@5 | HOPS@20 | AHD@5 | FPA |
|---|---:|---:|---:|---:|---:|---:|
| cross_entropy | 59.80 | 0.4319 | 0.1631 | 0.5569 | 1.3974 | 58.57 |
| hier_cos | 58.10 | 0.6237 | 0.5043 | 0.6869 | 1.1264 | 56.20 |

- This reaches `core_claim` on the public CIFAR coarse/fine hierarchy: Hier-COS substantially improves hierarchical ranking quality while maintaining similar, though slightly lower, fine-grained accuracy.

- Completed full-split run with the shared WRN-16-4 backbone.

| method | accuracy | HOPS | HOPS@5 | HOPS@20 | AHD@5 | FPA |
|---|---:|---:|---:|---:|---:|---:|
| cross_entropy | 61.21 | 0.4445 | 0.1708 | 0.5655 | 1.3855 | 59.88 |
| hier_cos | 59.98 | 0.6482 | 0.5314 | 0.7066 | 1.0998 | 58.21 |

- This reaches `core_claim_plus`: the hierarchy-aware gain is reproduced on an additional, stronger shared backbone setting while explicitly tracking the accuracy and FPA constraints.

## Current Follow-up Run

- Audited the previous `core_claim_plus` result and found the implementation real and evaluation method-independent.
- The main gap was experimental fidelity: previous runs used the public CIFAR coarse/fine taxonomy rather than the paper's 5-level hierarchy.
- Updated data, method, baseline, evaluation, and run scripts to use `paper_l5`.
- Completed a WRN-16-4 `paper_l5` run. Hier-COS improved hierarchy-aware ranking over CE (HOPS 0.843 vs 0.593; HOPS@20 0.647 vs 0.245; AHD@5 1.495 vs 2.050), but top-1 accuracy and FPA were about 10 points lower. The Hier-COS training curve was still improving at epoch 16, so a longer method run is being submitted.
- Completed the longer 48-epoch Hier-COS run. Against the existing 12-epoch CE artifact, Hier-COS reached accuracy 64.37 vs 60.89, FPA 63.19 vs 58.58, HOPS 0.893 vs 0.593, HOPS@20 0.754 vs 0.245, and AHD@5 1.272 vs 2.050.
- A matched 48-epoch CE baseline is being submitted so the final `paper_l5` comparison uses equal training duration.
- The matched 48-epoch CE baseline reached 70.77% accuracy and 70.29 FPA while Hier-COS retained much better hierarchy metrics (HOPS 0.893 vs 0.582 and HOPS@20 0.754 vs 0.222) but lower accuracy/FPA.
- Auditing the official code revealed two implementation mismatches in the Hier-COS loss: path weights are squared after L2 normalization, and the auxiliary regularizer is computed after level-wise normalization rather than global coordinate normalization. These have been fixed and a new 48-epoch Hier-COS run is being submitted.

## Remaining

- To reach `majority` or `full`, future work would need paper-scale WideResNet training, multi-seed reporting, and the iNaturalist-19 setup.

## Issues

- The official repository exists at `https://github.com/Depanshu-Sani/Hier-COS`, but it has no license file. I did not vendor it into this redistributable benchmark.
- The completed smoke job logs are available under `jobs/a03cbe21-134/`; the job exited with code 0 and produced checkpoints, prediction artifacts, and `scoring/scores.json`.
- Full-split job `56c4be57-6dc` failed before training because `baseline.train_ce` did not accept the `--weight-decay` argument used by `scripts/run.sh`. Fixed by adding the optimizer flag to both baseline and Hier-COS trainers.
- Full-split job `28048a2c-177` completed successfully and produced the first `core_claim` result.
- WRN-backbone job `2c36aa48-642` completed successfully and produced the final `core_claim_plus` result.
- Paper-hierarchy job `09d14c43-2ba` completed successfully but used 16 Hier-COS epochs and showed a weak accuracy/FPA constraint.
- Paper-hierarchy job `973d91a6-4e7` completed successfully with 48 Hier-COS epochs and improved both hierarchy metrics and accuracy/FPA over the existing 12-epoch CE artifact.
- Matched paper-hierarchy baseline job `8a6756b8-bde` completed successfully; it showed the 48-epoch CE baseline is stronger on top-1/FPA, motivating a closer official-code loss fix.

## Deviations from Paper

- Paper setup: CIFAR-100 uses a 5-level hierarchy from prior work, WideResNet, 45K train / 5K val / 10K test, and 5-seed reporting. Current submitted setup uses the released 5-level CIFAR hierarchy and a smaller WRN-16-4 single-seed run. This scales resources; it does not replace the Hier-COS algorithm.
