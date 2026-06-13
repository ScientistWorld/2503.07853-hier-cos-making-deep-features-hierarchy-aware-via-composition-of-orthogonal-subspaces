# Scientist Constraints

## Data

Use the CIFAR-100 train/test split obtained by `scripts/download.sh`. Do not add external labeled images, web images, or alternate CIFAR labels. The test split must be used only for final evaluation.

## Hierarchy

Use the taxonomy exposed by `data/cifar100_hierarchy.py` for this compact benchmark. Changing the tree changes HOPS, AHD, MS, and FPA and invalidates comparison.

## Model Scale

Use a single image classifier with no ensemble. Keep the backbone scale within the same order as the provided small CIFAR CNN unless explicitly comparing to the paper-scale reference in a separate experiment. Improvements should come from hierarchy-aware methodology, not a much larger model.

## Evaluation

Write predictions in the same `.npz` format and evaluate with `scripts/evaluate.sh`. Do not edit `eval/` or `data/` to favor a method.

## Compliance Checklist

- [ ] CIFAR-100 data comes from `scripts/download.sh`
- [ ] No external labeled training data used
- [ ] Test split not used for training or model selection
- [ ] Taxonomy in `data/cifar100_hierarchy.py` unchanged
- [ ] Single model, no ensemble
- [ ] Scores produced by `scripts/evaluate.sh`
