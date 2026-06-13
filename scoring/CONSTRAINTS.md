# Scientist Constraints

## Data

Use the CIFAR-100 train/test split obtained by `scripts/download.sh`. Do not add external labeled images, web images, or alternate CIFAR labels. The test split must be used only for final evaluation.

## Hierarchy

Use the `paper_l5` taxonomy exposed by `data/cifar100_hierarchy.py` for the primary benchmark. Changing the tree changes HOPS, AHD, MS, and FPA and invalidates comparison unless the result is clearly labeled as a hierarchy ablation.

## Model Scale

Use a single image classifier with no ensemble. Keep the backbone scale within the same order as a CIFAR WideResNet-style classifier unless explicitly labeling a model-scale ablation. Improvements should come from hierarchy-aware methodology, not a much larger model.

## Evaluation

Write predictions in the same `.npz` format and evaluate with `scripts/evaluate.sh`. Do not edit `eval/` or `data/` to favor a method.

## Compliance Checklist

- [ ] CIFAR-100 data comes from `scripts/download.sh`
- [ ] No external labeled training data used
- [ ] Test split not used for training or model selection
- [ ] `paper_l5` taxonomy in `data/cifar100_hierarchy.py` unchanged for the primary benchmark
- [ ] Single model, no ensemble
- [ ] Scores produced by `scripts/evaluate.sh`
