#!/bin/bash
# Reproduce the reference end to end inside the container (no internet):
# train the flat cross-entropy baseline and Hier-COS (seed 7, paper recipe,
# best-validation checkpoint), freeze the baseline's raw outputs into the
# evaluators, then score the reference on the visible split and on the hidden
# test. Data must already be in data/downloads/ (scripts/download.sh).
# The hardening study that produced HARDENING.md is redteam/hardening/run_hardening.sh.
set -euo pipefail
cd /home/user
FREEZE_BASELINE=1 bash scripts/reproduce.sh
rm -f scoring/scores_train.json scoring/scores_test.json
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
