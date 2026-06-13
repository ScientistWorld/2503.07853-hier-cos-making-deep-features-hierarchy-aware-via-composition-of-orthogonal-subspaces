from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.cifar100_hierarchy import FINE_TO_COARSE, Cifar100Hierarchy, build_hierarchy


SLICE_NAME = "train"
SPLIT_SEED = 250307853
EXPERIMENT = "cifar100_hierarchical_classification"


def evaluate_prediction_file(path: Path) -> tuple[str, dict[str, float], dict[str, object]]:
    loaded = np.load(path, allow_pickle=True)
    labels = loaded["labels"].astype(int)
    indices = stratified_split_indices(labels, SLICE_NAME)
    arrays = slice_npz_arrays(loaded, len(labels), indices)
    method_key = method_key_from_arrays(arrays)
    metrics = evaluate_prediction_arrays(arrays)
    meta = {
        "file": str(path),
        "num_examples": int(len(indices)),
        "slice": SLICE_NAME,
    }
    return method_key, metrics, meta


def slice_npz_arrays(obj: np.lib.npyio.NpzFile, n_examples: int, indices: np.ndarray) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for key in obj.files:
        value = np.asarray(obj[key])
        if value.ndim > 0 and value.shape[0] == n_examples:
            arrays[key] = value[indices]
        else:
            arrays[key] = value
    return arrays


def stratified_split_indices(labels: np.ndarray, slice_name: str) -> np.ndarray:
    selected: list[np.ndarray] = []
    for cls in sorted(np.unique(labels).astype(int).tolist()):
        cls_indices = np.flatnonzero(labels == cls)
        rng = np.random.default_rng(SPLIT_SEED + cls)
        shuffled = cls_indices[rng.permutation(len(cls_indices))]
        midpoint = len(shuffled) // 2
        if slice_name == "train":
            chosen = shuffled[:midpoint]
        elif slice_name == "test":
            chosen = shuffled[midpoint:]
        else:
            raise ValueError(f"unknown slice: {slice_name}")
        selected.append(chosen)
    return np.sort(np.concatenate(selected)).astype(int)


def evaluate_prediction_arrays(obj: dict[str, np.ndarray]) -> dict[str, float]:
    hierarchy_name = scalar_str(obj["hierarchy"]) if "hierarchy" in obj else "public"
    hierarchy = build_hierarchy(hierarchy_name)
    labels = obj["labels"].astype(int)
    leaf_scores = obj["leaf_scores"].astype(float)
    pred = leaf_scores.argmax(axis=1)
    top_order = np.argsort(-leaf_scores, axis=1)

    coarse_labels = obj["coarse_labels"].astype(int) if "coarse_labels" in obj else np.array(FINE_TO_COARSE)[labels]
    if "coarse_scores" in obj:
        coarse_pred = obj["coarse_scores"].astype(float).argmax(axis=1)
    else:
        coarse_pred = np.array(FINE_TO_COARSE)[pred]

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

    return {
        "accuracy": float((pred == labels).mean() * 100.0),
        "ms": ms,
        "ahd@1": float(dists_top.mean()),
        "ahd@5": ahd_at_k(labels, top_order, 5, distance_matrix),
        "ahd@20": ahd_at_k(labels, top_order, 20, distance_matrix),
        "hops": hops(labels, top_order, None, distance_matrix),
        "hops@5": hops(labels, top_order, 5, distance_matrix),
        "hops@20": hops(labels, top_order, 20, distance_matrix),
        "fpa": fpa,
    }


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


def method_key_from_arrays(obj: dict[str, np.ndarray]) -> str:
    method_name = scalar_str(obj["method_name"])
    name_map = {
        "baseline_ce": "cross_entropy",
        "hier_cos": "hier_cos",
        "hier_cos_smoke": "hier_cos",
        "baseline_ce_smoke": "cross_entropy",
    }
    return name_map.get(method_name, method_name)


def scalar_str(value: np.ndarray) -> str:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return str(arr.item())
    return str(arr)


def evaluate_all(prediction_dir: Path, output_path: Path) -> dict[str, object]:
    results: dict[str, dict[str, float]] = {}
    per_method_meta: dict[str, dict[str, object]] = {}
    hierarchy_name = "public"
    for npz in sorted(prediction_dir.glob("*.npz")):
        method_key, metrics, meta = evaluate_prediction_file(npz)
        if method_key in results:
            raise ValueError(f"duplicate method key {method_key!r} from {npz}")
        with np.load(npz, allow_pickle=True) as loaded:
            hierarchy_name = scalar_str(loaded["hierarchy"]) if "hierarchy" in loaded else hierarchy_name
        results[method_key] = metrics
        per_method_meta[method_key] = meta

    hierarchy = build_hierarchy(hierarchy_name)
    scores = {
        "experiments": {
            EXPERIMENT: {
                "results": results,
                "_meta": {
                    "hierarchy": hierarchy.name,
                    "num_nodes": hierarchy.num_nodes,
                    "prediction_dir": str(prediction_dir),
                    "slice": SLICE_NAME,
                    "split_strategy": "stratified_by_fine_label_50_50_complement",
                    "split_seed": SPLIT_SEED,
                    "methods": per_method_meta,
                },
            }
        }
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scores, indent=2, sort_keys=True))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", default="outputs/predictions")
    parser.add_argument("--output", default="scoring/scores_train.json")
    args = parser.parse_args()
    scores = evaluate_all(Path(args.prediction_dir), Path(args.output))
    print(json.dumps(scores, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
