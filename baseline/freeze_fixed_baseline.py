"""Freeze the cross-entropy baseline's raw outputs into the evaluators.

Gym-builder step (run once by scripts/reproduce.sh with FREEZE_BASELINE=1):
the scored ``cross_entropy`` row is recomputed by each evaluator from its own
frozen copy, so replacing or deleting baseline artifacts on disk cannot move
the comparison. eval/train/ receives only the validation rows, eval/test/ only
the test rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="outputs/baseline/cross_entropy.npz")
    parser.add_argument("--train-dest", default="eval/train/fixed/cross_entropy.npz")
    parser.add_argument("--test-dest", default="eval/test/fixed/cross_entropy.npz")
    args = parser.parse_args()
    with np.load(args.source, allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    for prefix, dest in (("val_", args.train_dest), ("test_", args.test_dest)):
        subset = {key: value.astype(np.float32) for key, value in arrays.items() if key.startswith(prefix)}
        if not subset:
            raise SystemExit(f"{args.source} has no {prefix}* arrays")
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, **subset)
        print(f"froze {sorted(subset)} -> {dest}")


if __name__ == "__main__":
    main()
