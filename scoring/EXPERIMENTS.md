# Experiment Overview

## cifar100_hierarchical_classification

Tests the central hierarchical image classification setting on CIFAR-100 with a fixed 5-level class taxonomy. The question is whether a classifier's full ranked prediction respects semantic similarity (wrong labels that are taxonomically close should be ranked above distant ones) while keeping fine-grained top-1 and top-5 recognition, and whether its predictions at every taxonomy level form a consistent, correct path.

Benefit side: ranking quality against the taxonomy-induced preference order (HOPS over the full ranking and its top-5 / top-20 restrictions), plus the severity of mistakes (mistake severity and average hierarchical distance at 1, 5 and 20).

Constraint side: fine-grained top-1 and top-5 accuracy, accuracy at the level just above the leaves, full-path accuracy, and path consistency. A method cannot gain by trading away recognition or multi-level correctness.

The comparison point is a flat cross-entropy classifier of the same scale, trained on the same split with the same recipe. Development scores come from a visible 5,000-image validation split carved from the training set. Final scores come from the 10,000 official test images, whose labels are withheld.
