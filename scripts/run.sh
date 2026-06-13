#!/bin/bash
# Example job script submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Make sure all data, code, and dependencies are already in the
# workspace or baked into the container before submitting.

set -e

cd /home/user

python - <<'PY'
from pathlib import Path
required = Path("data/downloads/cifar-100-python/train")
if not required.exists():
    raise SystemExit("CIFAR-100 is missing. Run scripts/download.sh before submitting.")
PY

mkdir -p outputs/predictions checkpoints scoring

rm -f outputs/predictions/baseline_ce.npz outputs/predictions/hier_cos.npz

python -m baseline.train_ce \
  --run-name baseline_ce \
  --epochs "${EPOCHS:-100}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --lr "${LR:-0.1}" \
  --weight-decay 0.0005 \
  --optimizer sgd \
  --backbone official_wrn28_8 \
  --hierarchy paper_l5

python -m method.train_hiercos \
  --run-name hier_cos \
  --epochs "${EPOCHS:-100}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --lr "${LR:-0.1}" \
  --weight-decay 0.0005 \
  --optimizer sgd \
  --backbone official_wrn28_8 \
  --alpha "${ALPHA:-0.05}" \
  --hierarchy paper_l5

bash scripts/evaluate.sh outputs/predictions
python validate.py --compare
