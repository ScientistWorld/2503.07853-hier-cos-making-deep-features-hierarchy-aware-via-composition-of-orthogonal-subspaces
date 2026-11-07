#!/bin/bash
set -euo pipefail
cd /home/user
if [ ! -f redteam/reference_checkpoint/artifacts/hier_cos.npz ]; then
  python redteam/train_variant.py checkpoint --gpu "${AUDIT_GPU:-0}" > redteam/reference_checkpoint/train.log 2>&1
fi
python redteam/run_case.py reference_checkpoint
