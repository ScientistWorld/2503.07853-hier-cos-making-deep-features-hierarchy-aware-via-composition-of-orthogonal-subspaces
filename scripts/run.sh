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

rm -rf outputs/predictions
mkdir -p outputs/predictions checkpoints scoring

python -m baseline.train_ce \
  --run-name baseline_ce \
  --epochs 12 \
  --batch-size 256 \
  --lr 0.001 \
  --weight-decay 0.0005 \
  --backbone wrn16_4 \
  --hierarchy paper_l5

python -m method.train_hiercos \
  --run-name hier_cos \
  --epochs 16 \
  --batch-size 256 \
  --lr 0.001 \
  --weight-decay 0.0005 \
  --backbone wrn16_4 \
  --alpha 0.05 \
  --hierarchy paper_l5

bash scripts/evaluate.sh outputs/predictions
python validate.py --compare
