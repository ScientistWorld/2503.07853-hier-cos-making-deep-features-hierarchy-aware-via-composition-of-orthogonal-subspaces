#!/bin/bash
# Run the ResearchGym train/test evaluation split over existing final predictions.

set -euo pipefail

cd /home/user

python - <<'PY'
from pathlib import Path

prediction_dir = Path("outputs/predictions")
required = [
    prediction_dir / "baseline_ce.npz",
    prediction_dir / "hier_cos.npz",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(
        "Missing final prediction artifact(s): " + ", ".join(missing)
    )
PY

mkdir -p scoring
rm -f scoring/scores_train.json scoring/scores_test.json

bash scripts/evaluate_train.sh outputs/predictions
bash scripts/evaluate_test.sh outputs/predictions
