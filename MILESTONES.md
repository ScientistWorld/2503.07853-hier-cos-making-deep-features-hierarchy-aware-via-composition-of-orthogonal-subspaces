# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

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

## Stop Justification
- Stopped at `core_claim_plus`. The reusable environment reproduces the core hierarchy-aware effect on two full-split settings, including a WideResNet-family backbone. Remaining gaps to `majority` or `full` are the paper's proprietary 5-level CIFAR hierarchy asset, full WideResNet paper scale, multi-seed reporting, and iNaturalist experiments.
