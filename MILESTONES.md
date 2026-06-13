# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-06-13 20:31] - none
- Read the paper and extracted the Hier-COS algorithm and HOPS metric.
- Implemented the initial redistributable CIFAR-100 reproduction scaffold.
- Prepared a first compute job to test whether the method runs end-to-end.

### [2026-06-13 20:46] - method_runs
- Completed a GPU smoke job on 6,000 CIFAR-100 training examples and 2,000 test examples.
- Verified that both the Cross Entropy baseline and the Hier-COS implementation train, write prediction artifacts, and score through `scripts/evaluate.sh`.
- Smoke result does not yet support the core claim: Hier-COS improved HOPS@5 over the baseline, but accuracy and several hierarchy metrics remained weak.

## Stop Justification
- Not stopped.
