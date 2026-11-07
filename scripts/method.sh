#!/bin/bash
# Run the paper's proposed method (Hier-COS) with the paper's CIFAR-100
# configuration (defaults in method/train_hiercos.py), writing
# outputs/predictions/<run-name>.npz, then score it on the visible split.
#
# Usage:
#   bash scripts/method.sh [--seed 7] [--run-name hier_cos] [...]

set -e

cd /home/user

python -m method.train_hiercos "$@"
bash scripts/evaluate.sh outputs/predictions
