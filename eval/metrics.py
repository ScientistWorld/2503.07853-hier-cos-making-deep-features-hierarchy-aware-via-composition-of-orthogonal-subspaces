from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data.cifar100_hierarchy import Cifar100Hierarchy, build_hierarchy


def evaluate_prediction_file(path: Path) -> dict[str, float]:
    obj = np.load(path, allow_pickle=True)
    hierarchy_name = str(obj["hierarchy"]) if "hierarchy" in obj else "public"
    hierarchy = build_hierarchy(hierarchy_name)
    labels = obj["labels"].astype(int)
    leaf_scores = obj["leaf_scores"].astype(float)
    pred = leaf_scores.argmax(axis=1)
    top_order = np.argsort(-leaf_scores, axis=1)

    coarse_level = max(0, len(hierarchy.level_node_indices) - 2)
    hierarchy_level_labels = np.array(hierarchy.fine_to_level_labels, dtype=int)
    coarse_labels = hierarchy_level_labels[labels, coarse_level]
    coarse_scores = explicit_coarse_scores(obj, hierarchy)
    has_coarse_scores = coarse_scores is not None
    if coarse_scores is not None:
        coarse_pred = coarse_scores.argmax(axis=1)
    else:
        coarse_pred = hierarchy_level_labels[pred, coarse_level]

    distance_matrix = lca_distance_matrix(hierarchy)
    dists_top = distance_matrix[labels, pred]
    wrong = pred != labels
    ms = float(dists_top[wrong].mean()) if wrong.any() else 0.0
    if "level_labels" in obj and any(f"level_scores_{idx}" in obj for idx in range(len(hierarchy.level_node_indices))):
        level_labels = obj["level_labels"].astype(int)
        correct_path = np.ones(labels.shape[0], dtype=bool)
        for idx in range(len(hierarchy.level_node_indices)):
            key = f"level_scores_{idx}"
            if key in obj:
                correct_path &= obj[key].astype(float).argmax(axis=1) == level_labels[:, idx]
        fpa = float(correct_path.mean() * 100.0)
    else:
        fpa = float(((pred == labels) & (coarse_pred == coarse_labels)).mean() * 100.0)

    metrics = {
        "accuracy": float((pred == labels).mean() * 100.0),
        "macro_accuracy": macro_accuracy(labels, pred, leaf_scores.shape[1]),
        "top5_accuracy": topk_accuracy(labels, top_order, 5),
        "coarse_accuracy": float((coarse_pred == coarse_labels).mean() * 100.0),
        "fine_coarse_consistency": fine_coarse_consistency(coarse_pred, pred, hierarchy) if has_coarse_scores else 0.0,
        "ms": ms,
        "ahd@1": float(dists_top.mean()),
        "ahd@5": ahd_at_k(labels, top_order, 5, distance_matrix),
        "ahd@20": ahd_at_k(labels, top_order, 20, distance_matrix),
        "hops": hops(labels, top_order, None, distance_matrix),
        "hops@5": hops(labels, top_order, 5, distance_matrix),
        "hops@20": hops(labels, top_order, 20, distance_matrix),
        "fpa": fpa,
    }
    return metrics


def macro_accuracy(labels: np.ndarray, pred: np.ndarray, num_classes: int) -> float:
    values = []
    for cls in range(num_classes):
        mask = labels == cls
        if mask.any():
            values.append(float((pred[mask] == labels[mask]).mean()))
    return float(np.mean(values) * 100.0) if values else 0.0


def topk_accuracy(labels: np.ndarray, top_order: np.ndarray, k: int) -> float:
    k = min(k, top_order.shape[1])
    hits = [int(label) in row for label, row in zip(labels, top_order[:, :k])]
    return float(np.mean(hits) * 100.0) if hits else 0.0


def explicit_coarse_scores(obj: np.lib.npyio.NpzFile, hierarchy: Cifar100Hierarchy) -> np.ndarray | None:
    coarse_level = max(0, len(hierarchy.level_node_indices) - 2)
    key = f"level_scores_{coarse_level}"
    expected_width = len(hierarchy.level_node_indices[coarse_level])
    if key in obj and obj[key].shape[1] == expected_width:
        return obj[key].astype(float)
    if "coarse_scores" in obj and obj["coarse_scores"].shape[1] == expected_width:
        return obj["coarse_scores"].astype(float)
    return None


def fine_coarse_consistency(coarse_pred: np.ndarray, fine_pred: np.ndarray, hierarchy: Cifar100Hierarchy) -> float:
    coarse_level = max(0, len(hierarchy.level_node_indices) - 2)
    fine_parent = np.array(hierarchy.fine_to_level_labels, dtype=int)[fine_pred, coarse_level]
    return float((coarse_pred == fine_parent).mean() * 100.0)


def lca_distance_matrix(hierarchy: Cifar100Hierarchy) -> np.ndarray:
    distances = np.zeros((100, 100), dtype=float)
    for i, path_i in enumerate(hierarchy.fine_to_path):
        for j, path_j in enumerate(hierarchy.fine_to_path):
            common_depth = -1
            for depth, (node_i, node_j) in enumerate(zip(path_i, path_j)):
                if node_i != node_j:
                    break
                common_depth = depth
            distances[i, j] = len(path_i) - 1 - common_depth
    return distances


def ahd_at_k(labels: np.ndarray, top_order: np.ndarray, k: int, distance_matrix: np.ndarray) -> float:
    k = min(k, top_order.shape[1])
    vals = []
    for y, order in zip(labels, top_order[:, :k]):
        vals.extend(distance_matrix[int(y), order])
    return float(np.mean(vals))


def hops(labels: np.ndarray, top_order: np.ndarray, k: int | None, distance_matrix: np.ndarray) -> float:
    values = []
    for y, order in zip(labels, top_order):
        desired = desired_rank_vector(int(y), distance_matrix)
        rank_lookup = desired_rank_lookup(int(y), distance_matrix)
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


def desired_rank_lookup(true_label: int, distance_matrix: np.ndarray) -> dict[int, int]:
    distances = distance_matrix[true_label].tolist()
    non_empty = sorted(set(distances))
    distance_to_rank = {dist: rank for rank, dist in enumerate(non_empty)}
    return {cls: distance_to_rank[dist] for cls, dist in enumerate(distances)}


def desired_rank_vector(true_label: int, distance_matrix: np.ndarray) -> list[int]:
    lookup = desired_rank_lookup(true_label, distance_matrix)
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
    results = {}
    name_map = {
        "baseline_ce": "cross_entropy",
        "hier_cos": "hier_cos",
        "hier_cos_smoke": "hier_cos",
        "baseline_ce_smoke": "cross_entropy",
    }
    for npz in sorted(prediction_dir.glob("*.npz")):
        loaded = np.load(npz, allow_pickle=True)
        method_name = str(loaded["method_name"])
        hierarchy_name = str(loaded["hierarchy"]) if "hierarchy" in loaded else "public"
        method_key = name_map.get(method_name, method_name)
        results[method_key] = evaluate_prediction_file(npz)

    hierarchy = build_hierarchy(hierarchy_name if results else "public")
    scores = {
        "experiments": {
            "cifar100_hierarchical_classification": {
                "results": results,
                "_meta": {
                    "hierarchy": hierarchy.name,
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
