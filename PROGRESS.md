# Progress

## What Works

- Paper read and core algorithm identified: fixed orthonormal node bases, composed ancestor/self/descendant subspaces, weighted path KL loss, sparsity regularizer, and projection-norm inference.
- Implemented CIFAR-100 data loading, a public CIFAR hierarchy, Hier-COS training, a Cross Entropy baseline, and a method-independent evaluator.
- Added a selectable CIFAR WideResNet-style backbone so CE and Hier-COS can be compared on a backbone family closer to the paper's setup.

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

## Remaining

- To reach `majority` or `full`, future work would need the exact 5-level CIFAR hierarchy used by the paper, paper-scale WideResNet training, multi-seed reporting, and the iNaturalist-19 setup.

## Issues

- The official repository exists at `https://github.com/Depanshu-Sani/Hier-COS`, but it has no license file. I did not vendor it into this redistributable benchmark.
- The completed smoke job logs are available under `jobs/a03cbe21-134/`; the job exited with code 0 and produced checkpoints, prediction artifacts, and `scoring/scores.json`.
- Full-split job `56c4be57-6dc` failed before training because `baseline.train_ce` did not accept the `--weight-decay` argument used by `scripts/run.sh`. Fixed by adding the optimizer flag to both baseline and Hier-COS trainers.
- Full-split job `28048a2c-177` completed successfully and produced the first `core_claim` result.
- WRN-backbone job `2c36aa48-642` completed successfully and produced the final `core_claim_plus` result.

## Deviations from Paper

- Paper setup: CIFAR-100 uses a 5-level hierarchy from prior work, WideResNet, 45K train / 5K val / 10K test, and 5-seed reporting. Current compact setup: public CIFAR-100 20-coarse/100-fine hierarchy and single-seed runs. This scales resources and uses a public hierarchy asset; it does not replace the Hier-COS algorithm.
