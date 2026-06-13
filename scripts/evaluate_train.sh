#!/bin/bash
# Visible train-slice evaluator for future method development.
#
# Mirrors scripts/evaluate.sh's contract: the first argument is the directory
# containing method prediction .npz files. The script reads those artifacts and
# writes scoring/scores_train.json.

set -e

cd /home/user

mkdir -p scoring

python -m eval.train.evaluate --prediction-dir "${1:-outputs/predictions}" --output scoring/scores_train.json
