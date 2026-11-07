# Scientist Constraints

## Data

Use the CIFAR-100 files written by `scripts/download.sh`. Train only on the 45,000-image training split, the complement of `data/cifar100_val_split.json`. The 5,000 validation images are for development and model selection, not training. Do not add external labeled images or alternate CIFAR labels. Never obtain or use CIFAR-100 test labels from any source, including the public archive, mirrors, or any local cache. Use the unlabelled test images only to produce the final `test_*` scores: no training, fine-tuning, self-training, normalisation statistics or model selection on them.

## Hierarchy

Use the `paper_l5` taxonomy. The evaluators keep their own copy, so editing `data/cifar100_hierarchy.py` changes only your own method.

## Model Scale

Use a single image classifier with no ensembles and no multi-checkpoint averaging. Keep the backbone within the same order of size as the CIFAR WideResNet-28-8 used by the reference, with no external pretraining unless the result is clearly labelled as an ablation. Improvements should come from hierarchy-aware methodology, not a much larger model.

## Evaluation

Write one `.npz` per method in `outputs/predictions/` in the format of `briefing/evaluation.md`, and score with `scripts/evaluate.sh`. Do not edit `eval/` or `scripts/evaluate*.sh`. Artifacts carry only raw scores. Anything else written into them is ignored.

## Compliance Checklist

- [ ] CIFAR-100 data comes from `scripts/download.sh`; no external labeled data
- [ ] Trained on the 45K split only; validation split not used for training
- [ ] No test labels from any source; test images used only for final inference
- [ ] `paper_l5` taxonomy for the primary benchmark
- [ ] Single model, no ensemble, comparable backbone scale
- [ ] Artifacts in the documented format, scored by `scripts/evaluate.sh`
