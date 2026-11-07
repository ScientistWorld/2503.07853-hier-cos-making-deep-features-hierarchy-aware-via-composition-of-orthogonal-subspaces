#!/bin/bash
# Train the paper's flat cross-entropy baseline (WRN-28-8, same recipe and
# 45K training split as the method). Writes outputs/baseline/cross_entropy.npz.
#
# The scored cross_entropy row in every scores file is recomputed from the
# evaluator-owned frozen copy of this artifact (eval/*/fixed/), which only
# scripts/reproduce.sh with FREEZE_BASELINE=1 regenerates.
#
# Usage:
#   bash scripts/baseline.sh [--seed 7] [...]

set -e

cd /home/user

python -m baseline.train_ce "$@"
