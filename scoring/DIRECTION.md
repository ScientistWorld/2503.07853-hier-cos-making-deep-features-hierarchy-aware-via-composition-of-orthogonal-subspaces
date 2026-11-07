# Research Direction

## Research Problem

This benchmark is about improving hierarchical image classification: ranked predictions should reflect a known semantic taxonomy while preserving fine-grained top-1 and top-5 accuracy, and multi-level predictions should be correct and consistent along a tree path.

## Core Contribution

The paper's contribution is a hierarchy-aware representation and classification rule that uses taxonomy structure directly in the feature space. Future work should improve how class hierarchy information shapes representations, ranking, or multi-level prediction. Known weak spots of the reference that leave room: its top-5 accuracy is well below a flat classifier's, its fine accuracy is only on par, and its full-ranking and top-k preference scores are well below their bounds.

## Approach Scope

Work should stay within hierarchy-aware single-model classifiers. It is in scope to change the hierarchy-aware representation, class scoring rule, regularization, loss weighting, optimization schedule, data augmentation, inference views of a single model, or transformation module, provided the method still uses the fixed taxonomy during learning or inference.

## Out of Bounds

- Ignoring the hierarchy and optimizing only flat top-1 accuracy.
- Replacing the task with retrieval, captioning, or another paradigm.
- Any use of test labels, or training on test images or the validation split.
- Changing the evaluation taxonomy, the evaluators, or the artifact interface.
- Ensembles, much larger or externally pretrained backbones, or external labeled data as the source of improvement.
