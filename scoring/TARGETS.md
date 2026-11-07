# Evaluation Targets

## Primary: cifar100_hierarchical_classification

Tests whether ranked predictions on CIFAR-100 follow the paper's 5-level semantic hierarchy while preserving fine-grained recognition and multi-level correctness. Scored on the 10,000 official test images (hidden); developed on the visible 5,000-image validation split.
- **Primary metric**: `hops` (higher is better)
- **Benefit metrics**: `hops`, `hops@5`, `hops@20` (higher); `ms`, `ahd@1`, `ahd@5`, `ahd@20` (lower)
- **Constraint metrics**: `accuracy`, `top5_accuracy`, `coarse_accuracy`, `fpa`, `path_consistency` (higher)
- **Fixed comparison**: evaluator-owned `cross_entropy` baseline row
- **Validity**: malformed, non-finite or implausible (accuracy >= 97%) artifacts score as failed
- **Run**: `bash scripts/evaluate.sh` (val) / `bash scripts/evaluate_test.sh` (hidden test, no arguments)
