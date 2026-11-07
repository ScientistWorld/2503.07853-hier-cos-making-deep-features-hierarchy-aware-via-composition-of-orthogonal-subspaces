#!/bin/bash
# Reproduce the paper's CIFAR-100 results end-to-end (inside the container,
# workspace at /home/user, data fetched beforehand by scripts/download.sh).
#
#   1. flat cross-entropy baseline  -> outputs/baseline/cross_entropy.npz
#   2. Hier-COS (paper method)      -> outputs/predictions/hier_cos.npz
#   3. FREEZE_BASELINE=1 only: copy the baseline's raw outputs into the
#      evaluators (eval/train/fixed, eval/test/fixed) — gym-builder step
#   4. score on the visible split (scoring/scores.json)
#
# Each training run is WRN-28-8, 100 epochs, ~25 min on one GPU.

set -e

cd /home/user

if [ ! -e data/downloads/cifar-100-python/train ] || [ ! -e data/downloads/cifar-100-python/test_images.u8 ]; then
  echo "CIFAR-100 is missing. Run scripts/download.sh on an internet-enabled node first." >&2
  exit 1
fi
SEED="${SEED:-7}"
mkdir -p outputs/predictions outputs/baseline
bash scripts/baseline.sh --seed "$SEED" --run-name cross_entropy --epochs "${BASELINE_EPOCHS:-100}"
python -m method.train_hiercos --seed "$SEED" --run-name hier_cos --epochs "${METHOD_EPOCHS:-100}"
if [ "${FREEZE_BASELINE:-0}" = "1" ]; then
  python -m baseline.freeze_fixed_baseline --source outputs/baseline/cross_entropy.npz
fi
bash scripts/evaluate.sh outputs/predictions
