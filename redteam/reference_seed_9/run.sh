#!/bin/bash
set -euo pipefail
cd /home/user
if [ ! -f redteam/reference_seed_9/artifacts/hier_cos.npz ]; then
  python redteam/train_variant.py seed --seed 9 --gpu "${AUDIT_GPU:-0}" > redteam/reference_seed_9/train.log 2>&1
fi
python redteam/run_case.py reference_seed_9
