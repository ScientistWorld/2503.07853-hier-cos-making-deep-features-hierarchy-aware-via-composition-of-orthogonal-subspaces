#!/bin/bash
# Reproduce the paper's results end-to-end.
#
# This script runs inside the compute container and should call
# scripts/baseline.sh and scripts/method.sh to run baselines and
# the paper's method, each of which calls scripts/evaluate.sh.
#
# Assumes:
#   - Container is built (environment/container.sif)
#   - Data and models are downloaded (via scripts/download.sh before running on a compute node)
#   - Working directory is /home/user
#
# This is the "gold standard" reproduction — 
# You do not need to run this, but keep this in good shape
# so other agents can easily run a full loop of reproduction and get the paper's numbers.

set -e

cd /home/user

python - <<'PY'
from pathlib import Path
required = Path("data/downloads/cifar-100-python/train")
if not required.exists():
    raise SystemExit("CIFAR-100 is missing. Run scripts/download.sh on an internet-enabled node first.")
PY
rm -rf outputs/predictions
mkdir -p outputs/predictions
bash scripts/baseline.sh --run-name baseline_ce --epochs "${BASELINE_EPOCHS:-100}" --batch-size "${BATCH_SIZE:-64}" --lr "${LR:-0.1}" --weight-decay 0.0005 --optimizer "${OPTIMIZER:-sgd}" --backbone "${BACKBONE:-official_wrn28_8}" --hierarchy "${HIERARCHY:-paper_l5}"
bash scripts/method.sh --run-name hier_cos --epochs "${METHOD_EPOCHS:-100}" --batch-size "${BATCH_SIZE:-64}" --lr "${LR:-0.1}" --weight-decay 0.0005 --optimizer "${OPTIMIZER:-sgd}" --backbone "${BACKBONE:-official_wrn28_8}" --alpha "${ALPHA:-0.05}" --hierarchy "${HIERARCHY:-paper_l5}"
bash scripts/evaluate.sh outputs/predictions
