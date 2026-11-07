"""Runnable shortcut implementations. Uses only scientist-visible inputs.

No hidden evaluator modules are imported. Prediction reuse is the audit's
explicitly permitted reuse of reference outputs for artifact-only checks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import re
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.cifar100_hierarchy import build_hierarchy

ORIGINAL = ROOT / "redteam/_original"
DEST = ROOT / "outputs/predictions"


def load(name="hier_cos"):
    with np.load(ORIGINAL / "predictions" / f"{name}.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def save(obj, name="hier_cos"):
    np.savez_compressed(DEST / f"{name}.npz", **obj)


def distance_table(h):
    # Rebuilt from the public taxonomy, independently of the evaluator.
    d = np.zeros((100, 100))
    for i, a in enumerate(h.fine_to_path):
        for j, b in enumerate(h.fine_to_path):
            common = 0
            for u, v in zip(a, b):
                if u != v:
                    break
                common += 1
            d[i, j] = len(a) - common
    return d


def oracle_scores(labels, hierarchy="paper_l5"):
    h = build_hierarchy(hierarchy)
    d = distance_table(h)
    # Lexicographic score: tree distance first; class index breaks only ties.
    scores = -d[labels] - np.arange(100)[None, :] * 1e-6
    paths = np.asarray(h.fine_to_level_labels)[labels]
    obj = dict(method_name="hier_cos", hierarchy=hierarchy, labels=labels,
               leaf_scores=scores, level_labels=paths)
    for level, nodes in enumerate(h.level_node_indices):
        obj[f"level_scores_{level}"] = np.eye(len(nodes))[paths[:, level]]
    return obj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("case")
    case = p.parse_args().case
    DEST.mkdir(parents=True, exist_ok=True)
    if case == "shortcut_test_labels":
        with (ROOT / "data/downloads/cifar-100-python/test").open("rb") as f:
            public_test = pickle.load(f, encoding="latin1")
        save(oracle_scores(np.asarray(public_test["fine_labels"], dtype=int)))
        return
    if case == "shortcut_constant_self_labels":
        # No images, model, labels, or hidden-split seed are needed.
        save(oracle_scores(np.zeros(10000, dtype=int)))
        return
    original = load()
    if case == "reference_artifacts":
        save(original)
    elif case.startswith("reference_seed_") or case in {
        "variant_more_training", "variant_bn_recalibration", "variant_less_training", "reference_checkpoint"
    }:
        shutil.copy2(ROOT / "redteam" / case / "artifacts/hier_cos.npz", DEST / "hier_cos.npz")
    elif case == "variant_flat_baseline":
        obj = load("baseline_ce")
        obj["method_name"] = "hier_cos"
        save(obj)
    elif case == "variant_taxonomy_rerank":
        obj = load("baseline_ce")
        h = build_hierarchy("paper_l5")
        d = distance_table(h)
        # Honest baseline: use flat classifier's predicted label and the known
        # taxonomy to rank alternatives. No ground truth is used for ranking.
        s = obj["leaf_scores"].astype(float)
        pred = s.argmax(1)
        order = np.argsort(np.argsort(s, axis=1), axis=1) / 100.
        obj["leaf_scores"] = -d[pred] + order * 1e-3
        obj["method_name"] = "hier_cos"
        save(obj)
    elif case == "shortcut_select_correct":
        n = len(original["labels"])
        keep = np.flatnonzero(original["leaf_scores"].argmax(1) == original["labels"])
        save({k: (v[keep] if np.ndim(v) and v.shape[0] == n else v)
              for k, v in original.items()})
    elif case == "shortcut_path_labels":
        h = build_hierarchy("paper_l5")
        original["level_labels"] = np.column_stack([
            original[f"level_scores_{i}"].argmax(1) for i in range(len(h.level_node_indices))])
        save(original)
    elif case == "shortcut_omit_leaf_path_level":
        # Submit only the coarse level. The FPA calculation no longer checks
        # the leaf prediction even though the leaf prediction is available.
        obj = {k: v for k, v in original.items()
               if not k.startswith("level_scores_") or k == "level_scores_3"}
        save(obj)
    elif case == "shortcut_baseline_replacement":
        save(original)
        obj = load("baseline_ce")
        obj["leaf_scores"] = np.zeros_like(obj["leaf_scores"])
        for k in list(obj):
            if k.startswith("level_scores_") or k in {"level_labels", "coarse_scores"}:
                del obj[k]
        save(obj, "baseline_ce")
    elif case == "shortcut_taxonomy_artifact":
        # Replace only the mutable data table, not the evaluator or Python
        # environment. All 100 labels alias the same path in the fixed artifact.
        path = ROOT / "data/cifar100_hierarchy.py"
        text = path.read_text()
        text, count = re.subn(r"PAPER_L5_ROWS = \[.*?\n\]", "PAPER_L5_ROWS = " +
                             repr([[0, 0, 0, 0, 0]] * 100), text, count=1, flags=re.S)
        assert count == 1
        path.write_text(text)
        n = len(original["labels"])
        obj = {k: v for k, v in original.items()
               if not k.startswith("level_scores_") and k not in {"level_labels", "coarse_scores"}}
        obj["level_labels"] = np.zeros((n, 5), dtype=int)
        for i in range(5):
            obj[f"level_scores_{i}"] = np.ones((n, 1))
        save(obj)
    elif case == "shortcut_write_scores":
        original.update(accuracy=np.array(100.), hops=np.array(1.), fpa=np.array(100.),
                        success=np.array(True), elapsed_seconds=np.array(-1.),
                        score=np.array(float("inf")))
        save(original)
    elif case in {"shortcut_constant", "shortcut_nan", "shortcut_infinity", "shortcut_empty"}:
        if case == "shortcut_empty":
            n = len(original["labels"])
            original = {k: (v[:0] if np.ndim(v) and v.shape[0] == n else v)
                        for k, v in original.items()}
        else:
            value = {"shortcut_constant": 0., "shortcut_nan": np.nan,
                     "shortcut_infinity": np.inf}[case]
            original["leaf_scores"] = np.full_like(original["leaf_scores"], value)
        save(original)
    elif case == "shortcut_method_alias":
        # Reuse the flat classifier and claim the proposed-method identifier.
        obj = load("baseline_ce")
        obj["method_name"] = "hier_cos"
        save(obj)
    else:
        raise ValueError(case)


if __name__ == "__main__":
    main()
