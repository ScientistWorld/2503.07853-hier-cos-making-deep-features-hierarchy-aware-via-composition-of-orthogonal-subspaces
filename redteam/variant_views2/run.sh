#!/bin/bash
set -euo pipefail
cd /home/user
if [ ! -f redteam/variant_views2/artifacts/hier_cos.npz ]; then
  python redteam/train_variant.py views2 --gpu "${AUDIT_GPU:-0}" > redteam/variant_views2/train.log 2>&1
fi
python redteam/run_case.py variant_views2
