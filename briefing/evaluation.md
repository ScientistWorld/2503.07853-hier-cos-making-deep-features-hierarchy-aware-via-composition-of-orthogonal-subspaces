# Evaluation

## Data protocol

CIFAR-100 with the paper's 5-level class taxonomy (`paper_l5` in `data/cifar100_hierarchy.py`; 2, 4, 8 and 20 nodes on the four non-leaf levels, 100 leaves).

| Split | Images | Labels | Use |
|---|---|---|---|
| train | 45,000 (official training set minus the validation split) | visible | training |
| val | 5,000 fixed images of the official training set, 50 per class (`data/cifar100_val_split.json`) | visible | development, scored by `scripts/evaluate.sh` |
| test | the 10,000 official test images (`data/downloads/cifar-100-python/test_images.u8`) | withheld | hidden final scoring |

`scripts/download.sh` writes the training pickle and the unlabelled test images. Both scored splits are exactly class-balanced.

## Prediction artifact

One file per method: `outputs/predictions/<method_key>.npz`. The file stem is the method's name in every score file; `cross_entropy` is reserved for the fixed baseline. Required arrays, all real-valued and finite, higher = more preferred:

| Array | Shape | Rows |
|---|---|---|
| `val_leaf_scores` | (5000, 100) | in `val_indices` order |
| `val_level_scores_{l}`, l = 0..3 | (5000, K_l), K = (2, 4, 8, 20) | same |
| `test_leaf_scores` | (10000, 100) | official test order |
| `test_level_scores_{l}`, l = 0..3 | (10000, K_l) | same |

Column j of `*_level_scores_{l}` is node j of level l (level 0 = children of the root, level 3 = parents of the leaves), i.e. value j in column `4 - l` of `PAPER_L5_ROWS`. `data/prediction_format.py` writes this format. Labels, paths, names, flags or numbers stored in the artifact are ignored: the evaluators own the ground truth, the taxonomy, the baseline and the split sizes.

A submission is scored as **failed** (worst value on every metric) if an array is missing, has the wrong shape or a non-real dtype, or contains NaN/inf. It also fails if its fine accuracy is at least 97%, which is beyond the CIFAR-100 label-noise ceiling and indicates label lookup.

## Metrics

Computed per split from the raw scores. The leaf ranking is the descending order of `*_leaf_scores`, and a level's prediction is the argmax of its `*_level_scores_{l}`. LCA distance between two leaves is the number of levels above the leaves up to their lowest common ancestor (0 to 5).

| Metric | Direction | Definition |
|---|---|---|
| `hops` (primary) | higher | Hierarchically Ordered Preference Score: weighted disagreement between the predicted ranking of all 100 classes and the taxonomy-induced preference order (true class, then classes by increasing LCA distance), normalised by the worst order. Weights halve at each new preference rank and decay linearly within a rank |
| `hops@5`, `hops@20` | higher | the same, counting only the top 5 / 20 ranked positions |
| `accuracy`, `top5_accuracy` | higher | fine-label top-1 / top-5 accuracy (%) |
| `coarse_accuracy` | higher | accuracy (%) at level 3 (20 nodes) |
| `fpa` | higher | full-path accuracy (%): top-1 leaf and all four level predictions correct |
| `path_consistency` | higher | % of examples whose four level predictions are exactly the ancestors of the top-1 leaf |
| `ms` | lower | mean LCA distance of top-1 mistakes (misclassified examples only) |
| `ahd@1`, `ahd@5`, `ahd@20` | lower | mean LCA distance of the top-1 / top-5 / top-20 ranked classes over all examples |

## Baselines and targets

Each score file has a `cross_entropy` row, recomputed by the evaluator from its own frozen copy of a flat cross-entropy classifier's outputs. That classifier is the same backbone, recipe and training split as the reference, with superclass scores formed as the sum of leaf probabilities. `scoring/reference.json` lists the paper's CIFAR-100 numbers (Table 2 and the CIFAR-100 part of Table 4) for this baseline and for the paper's best hierarchy-aware system.

## Protocol

Write the artifact, then run `bash scripts/evaluate.sh` (writes `scoring/scores.json` for the val split). `scripts/evaluate_train.sh` is the same val scoring written to `scoring/scores_train.json`. The hidden test (`scripts/evaluate_test.sh`, no arguments) applies identical code and rules to the `test_*` arrays of every artifact in `outputs/predictions/`. It never retrains or runs a model.
