#!/bin/bash
set -euo pipefail
cd /home/user
if [ ! -f redteam/reference_seed_7/artifacts/hier_cos.npz ]; then
  python redteam/train_variant.py seed --seed 7 --gpu "${AUDIT_GPU:-0}" > redteam/reference_seed_7/train.log 2>&1
fi
python redteam/run_case.py reference_seed_7
