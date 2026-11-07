#!/bin/bash
set -euo pipefail
cd /home/user
if [ ! -f redteam/variant_bn_recalibration/artifacts/hier_cos.npz ]; then
  python redteam/train_variant.py bn --gpu "${AUDIT_GPU:-0}" > redteam/variant_bn_recalibration/train.log 2>&1
fi
python redteam/run_case.py variant_bn_recalibration
