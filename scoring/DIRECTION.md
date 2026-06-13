# Research Direction

## Research Problem

This benchmark is about improving hierarchical image classification: ranked predictions should reflect a known semantic taxonomy while preserving fine-grained accuracy and path consistency.

## Core Contribution

The paper's contribution is a hierarchy-aware representation and classification rule that uses taxonomy structure directly in the feature space. Future work should improve how class hierarchy information shapes representations, ranking, or multi-level prediction.

## Approach Scope

Work should stay within hierarchy-aware single-model classifiers. It is in scope to change the hierarchy-aware representation, class scoring rule, regularization, loss weighting, optimization schedule, or transformation module, provided the method still explicitly uses the fixed taxonomy during learning or inference.

## Out of Bounds

- Ignoring the hierarchy and optimizing only flat top-1 accuracy.
- Replacing the task with retrieval, captioning, or another paradigm.
- Beating the score by changing the evaluation taxonomy or metric implementation.
- Using ensembles or external labeled data as the main source of improvement.
