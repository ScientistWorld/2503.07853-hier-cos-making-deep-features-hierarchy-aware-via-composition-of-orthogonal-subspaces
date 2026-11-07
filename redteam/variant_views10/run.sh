#!/bin/bash
set -euo pipefail
cd /home/user
if [ ! -f redteam/variant_views10/artifacts/hier_cos.npz ]; then
  python redteam/train_variant.py views10 --gpu "${AUDIT_GPU:-0}" > redteam/variant_views10/train.log 2>&1
fi
python redteam/run_case.py variant_views10
