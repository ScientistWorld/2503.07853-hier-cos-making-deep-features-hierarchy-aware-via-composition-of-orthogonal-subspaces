#!/bin/bash
# Visible evaluator for method development: the fixed 5,000-image validation
# split of the CIFAR-100 training set (data/cifar100_val_split.json).
#
# Scores every method artifact on disk (outputs/predictions/*.npz, keyed by
# file stem) plus the evaluator-owned frozen cross-entropy baseline, and writes
# scoring/scores_train.json. An optional first argument overrides the
# prediction directory.

set -e

cd /home/user

mkdir -p scoring

python -m eval.train.evaluate --prediction-dir "${1:-outputs/predictions}" --output scoring/scores_train.json
