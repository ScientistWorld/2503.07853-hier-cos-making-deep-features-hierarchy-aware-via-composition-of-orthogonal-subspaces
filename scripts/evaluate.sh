#!/bin/bash
# Standard evaluation for method development.
#
# Scores every prediction artifact in the given directory (default
# outputs/predictions/*.npz; format in briefing/evaluation.md) on the visible
# 5,000-image validation split, using only code in eval/ (never method/), and
# writes scoring/scores.json. The hidden test (scripts/evaluate_test.sh) uses
# the same artifact interface on the 10,000 official test images.
#
# Usage:
#   bash scripts/evaluate.sh [prediction_dir]

set -e

cd /home/user

mkdir -p scoring

python -m eval.train.evaluate --prediction-dir "${1:-outputs/predictions}" --output scoring/scores.json
