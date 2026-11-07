#!/bin/bash
# Hidden test evaluator (10,000 official CIFAR-100 test images).
#
# Scores every method artifact on disk (outputs/predictions/*.npz, one method
# per file, keyed by file stem) plus the evaluator-owned frozen cross-entropy
# baseline, and writes scoring/scores_test.json. Runs with no arguments; an
# optional first argument overrides the prediction directory.

set -e

cd /home/user

mkdir -p scoring

python -m eval.test.evaluate \
  --prediction-dir "${1:-outputs/predictions}" \
  --output scoring/scores_test.json
