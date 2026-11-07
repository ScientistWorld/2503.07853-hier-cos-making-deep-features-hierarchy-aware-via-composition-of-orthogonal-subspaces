#!/bin/bash
set -euo pipefail
cd /home/user
if [ ! -f redteam/variant_less_training/artifacts/hier_cos.npz ]; then
  python redteam/train_variant.py seed --seed 7 --gpu "${AUDIT_GPU:-0}" > redteam/variant_less_training/train.log 2>&1
fi
python redteam/run_case.py variant_less_training
