#!/bin/bash
# Held-out test-slice evaluator.
#
# Mirrors scripts/evaluate.sh's contract: the first argument is the directory
# containing method prediction .npz files. When scoring/scores_train.json exists,
# the test slice scores the same method keys that appeared on the train slice.

set -e

cd /home/user

mkdir -p scoring

python -m eval.test.evaluate \
  --prediction-dir "${1:-outputs/predictions}" \
  --output scoring/scores_test.json \
  --train-scores scoring/scores_train.json
