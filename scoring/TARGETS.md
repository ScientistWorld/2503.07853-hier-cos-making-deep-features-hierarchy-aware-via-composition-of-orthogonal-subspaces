# Evaluation Targets

## Primary: cifar100_hierarchical_classification

Tests whether ranked predictions on CIFAR-100 follow the semantic hierarchy while preserving fine-grained recognition.
- **Primary metric**: `hops` (higher is better)
- **Benefit metrics**: `hops`, `hops@5`, `hops@20`, `accuracy`
- **Constraint metrics**: `ms`, `ahd@1`, `ahd@5`, `ahd@20`, `fpa`
- **Run**: `bash scripts/evaluate.sh outputs/predictions`
