"""Prediction artifact format shared by every method (see briefing/evaluation.md).

One file per method: ``outputs/predictions/<method_key>.npz`` with

  val_leaf_scores          float (5000, 100)   rows in data/cifar100_val_split.json order
  val_level_scores_{l}     float (5000, K_l)   l = 0..3, K = (2, 4, 8, 20)
  test_leaf_scores         float (10000, 100)  rows in official CIFAR-100 test order
  test_level_scores_{l}    float (10000, K_l)

Higher score = more preferred. Column j of ``*_level_scores_{l}`` is node j of
taxonomy level l, i.e. the value j in column ``4 - l`` of
``PAPER_L5_ROWS`` (level 0 is the root's children, level 3 the 20 nodes just
above the leaves). Labels are never part of the artifact: evaluators own them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

NUM_CLASSES = 100
LEVEL_WIDTHS = (2, 4, 8, 20)
SPLIT_SIZES = {"val": 5000, "test": 10000}


def save_predictions(
    path: str | Path,
    val_leaf_scores: np.ndarray,
    val_level_scores: list[np.ndarray],
    test_leaf_scores: np.ndarray,
    test_level_scores: list[np.ndarray],
) -> Path:
    path = Path(path)
    arrays: dict[str, np.ndarray] = {}
    for split, leaf, levels in (
        ("val", val_leaf_scores, val_level_scores),
        ("test", test_leaf_scores, test_level_scores),
    ):
        n = SPLIT_SIZES[split]
        leaf = np.asarray(leaf, dtype=np.float32)
        if leaf.shape != (n, NUM_CLASSES):
            raise ValueError(f"{split}_leaf_scores must have shape {(n, NUM_CLASSES)}, got {leaf.shape}")
        arrays[f"{split}_leaf_scores"] = leaf
        if len(levels) != len(LEVEL_WIDTHS):
            raise ValueError(f"expected {len(LEVEL_WIDTHS)} level score arrays for {split}")
        for level, (scores, width) in enumerate(zip(levels, LEVEL_WIDTHS)):
            scores = np.asarray(scores, dtype=np.float32)
            if scores.shape != (n, width):
                raise ValueError(f"{split}_level_scores_{level} must have shape {(n, width)}, got {scores.shape}")
            arrays[f"{split}_level_scores_{level}"] = scores
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path
