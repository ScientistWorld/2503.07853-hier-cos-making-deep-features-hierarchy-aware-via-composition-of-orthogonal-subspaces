from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data.cifar100_hierarchy import FINE_TO_COARSE, build_hierarchy, lca_distance


def evaluate_prediction_file(path: Path) -> dict[str, float]:
    obj = np.load(path, allow_pickle=True)
    labels = obj["labels"].astype(int)
    leaf_scores = obj["leaf_scores"].astype(float)
    pred = leaf_scores.argmax(axis=1)
    top_order = np.argsort(-leaf_scores, axis=1)

    coarse_labels = obj["coarse_labels"].astype(int) if "coarse_labels" in obj else np.array(FINE_TO_COARSE)[labels]
    if "coarse_scores" in obj:
        coarse_pred = obj["coarse_scores"].astype(float).argmax(axis=1)
    else:
        coarse_pred = np.array(FINE_TO_COARSE)[pred]

    dists_top = np.vectorize(lca_distance)(labels, pred)
    wrong = pred != labels
    ms = float(dists_top[wrong].mean()) if wrong.any() else 0.0

    metrics = {
        "accuracy": float((pred == labels).mean() * 100.0),
        "ms": ms,
        "ahd@1": float(dists_top.mean()),
        "ahd@5": ahd_at_k(labels, top_order, 5),
        "ahd@20": ahd_at_k(labels, top_order, 20),
        "hops": hops(labels, top_order, None),
        "hops@5": hops(labels, top_order, 5),
        "hops@20": hops(labels, top_order, 20),
        "fpa": float(((pred == labels) & (coarse_pred == coarse_labels)).mean() * 100.0),
    }
    return metrics


def ahd_at_k(labels: np.ndarray, top_order: np.ndarray, k: int) -> float:
    k = min(k, top_order.shape[1])
    vals = []
    for y, order in zip(labels, top_order[:, :k]):
        vals.extend(lca_distance(int(y), int(p)) for p in order)
    return float(np.mean(vals))


def hops(labels: np.ndarray, top_order: np.ndarray, k: int | None) -> float:
    values = []
    for y, order in zip(labels, top_order):
        desired = desired_rank_vector(int(y))
        rank_lookup = desired_rank_lookup(int(y))
        predicted = np.array([rank_lookup[int(cls)] for cls in order], dtype=float)
        z = np.array(desired, dtype=float)
        eta = eta_weights(z)
        if k is None:
            worst = z[::-1]
            active_eta = eta
        else:
            kk = min(k, len(z))
            worst = z.copy()
            worst[:kk] = z[:kk][::-1]
            active_eta = eta.copy()
            active_eta[kk:] = 0.0
        denom = float(np.sum(active_eta * np.abs(z - worst)))
        if denom <= 0:
            score = 1.0 if predicted[0] == 0 else 0.0
        else:
            score = 1.0 - float(np.sum(active_eta * np.abs(z - predicted)) / denom)
        values.append(max(0.0, score) if k is not None else score)
    return float(np.mean(values))


def desired_rank_lookup(true_label: int) -> dict[int, int]:
    distances = [lca_distance(true_label, other) for other in range(100)]
    non_empty = sorted(set(distances))
    distance_to_rank = {dist: rank for rank, dist in enumerate(non_empty)}
    return {cls: distance_to_rank[dist] for cls, dist in enumerate(distances)}


def desired_rank_vector(true_label: int) -> list[int]:
    lookup = desired_rank_lookup(true_label)
    counts: dict[int, int] = {}
    for rank in lookup.values():
        counts[rank] = counts.get(rank, 0) + 1
    out: list[int] = []
    for rank in sorted(counts):
        out.extend([rank] * counts[rank])
    return out


def eta_weights(z: np.ndarray) -> np.ndarray:
    eta = np.zeros_like(z, dtype=float)
    start = 0
    while start < len(z):
        rank = int(z[start])
        end = start
        while end < len(z) and int(z[end]) == rank:
            end += 1
        count = end - start
        hi = 2.0 ** (-rank)
        lo = 2.0 ** (-(rank + 1))
        if count == 1:
            eta[start] = hi
        else:
            eta[start:end] = np.linspace(hi, lo, count, endpoint=False)
        start = end
    return eta


def evaluate_all(prediction_dir: Path, output_path: Path) -> dict:
    hierarchy = build_hierarchy()
    results = {}
    name_map = {
        "baseline_ce": "cross_entropy",
        "hier_cos": "hier_cos",
        "hier_cos_smoke": "hier_cos",
        "baseline_ce_smoke": "cross_entropy",
    }
    for npz in sorted(prediction_dir.glob("*.npz")):
        method_name = str(np.load(npz, allow_pickle=True)["method_name"])
        method_key = name_map.get(method_name, method_name)
        results[method_key] = evaluate_prediction_file(npz)

    scores = {
        "experiments": {
            "cifar100_hierarchical_classification": {
                "results": results,
                "_meta": {
                    "hierarchy": "public CIFAR-100 20-coarse/100-fine taxonomy",
                    "num_nodes": hierarchy.num_nodes,
                    "prediction_dir": str(prediction_dir),
                },
            }
        }
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scores, indent=2, sort_keys=True))
    return scores


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", default="outputs/predictions")
    parser.add_argument("--output", default="scoring/scores.json")
    args = parser.parse_args()
    scores = evaluate_all(Path(args.prediction_dir), Path(args.output))
    print(json.dumps(scores, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
