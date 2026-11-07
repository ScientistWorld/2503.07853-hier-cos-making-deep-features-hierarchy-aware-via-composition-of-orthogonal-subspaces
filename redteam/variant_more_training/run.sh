#!/bin/bash
set -euo pipefail
cd /home/user
if [ ! -f redteam/variant_more_training/artifacts/hier_cos.npz ]; then
  python redteam/train_variant.py extended --gpu "${AUDIT_GPU:-0}" > redteam/variant_more_training/train.log 2>&1
fi
python redteam/run_case.py variant_more_training
