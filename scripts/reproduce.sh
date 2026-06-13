#!/bin/bash
# Reproduce the paper's results end-to-end.
#
# This script runs inside the compute container and should call
# scripts/baseline.sh and scripts/method.sh to run baselines and
# the paper's method, each of which calls scripts/evaluate.sh.
#
# Assumes:
#   - Container is built (environment/container.sif)
#   - Data and models are downloaded (via scripts/download.sh)
#   - Working directory is /home/user
#
# This is the "gold standard" reproduction — 
# You do not need to run this, but keep this in good shape
# so other agents can easily run a full loop of reproduction and get the paper's numbers.

set -e

cd /home/user

bash scripts/download.sh
rm -rf outputs/predictions
mkdir -p outputs/predictions
bash scripts/baseline.sh --run-name baseline_ce --epochs "${EPOCHS:-12}" --batch-size "${BATCH_SIZE:-256}" --backbone "${BACKBONE:-wrn16_4}"
bash scripts/method.sh --run-name hier_cos --epochs "${EPOCHS:-16}" --batch-size "${BATCH_SIZE:-256}" --backbone "${BACKBONE:-wrn16_4}" --alpha "${ALPHA:-0.05}"
bash scripts/evaluate.sh outputs/predictions
