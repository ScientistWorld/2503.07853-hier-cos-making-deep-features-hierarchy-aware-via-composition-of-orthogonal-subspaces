"""Hidden test evaluator: the 10,000 official CIFAR-100 test images.

Self-contained by design. Everything on the evaluator's side of the comparison
is owned here and never read from a submission or from `data/`:

* ground truth: ``test_labels.json`` (integrity-checked by SHA-256);
* the paper's 5-level CIFAR-100 taxonomy: ``PAPER_L5_ROWS`` below;
* the flat cross-entropy baseline: raw outputs frozen in ``fixed/``;
* the test size, row order, metric definitions and validity rules.

Every ``*.npz`` in the prediction directory is one method, keyed by its file
stem. Only the raw score arrays ``test_leaf_scores`` and
``test_level_scores_{0..3}`` are read; labels, paths, names, flags or numbers
written by a method are ignored. A malformed submission (missing array, wrong
shape, non-numeric or non-finite value) or an implausible one (fine accuracy
at or above ``MAX_PLAUSIBLE_ACCURACY``, i.e. beyond the label-noise ceiling of
CIFAR-100, which indicates label lookup) is scored as failed with worst-case
values for every metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SPLIT = "test"
NUM_EXAMPLES = 10_000
NUM_CLASSES = 100
EXPERIMENT = "cifar100_hierarchical_classification"
HERE = Path(__file__).resolve().parent
LABELS_FILE = HERE / "test_labels.json"
LABELS_SHA256 = "c78feeaa17dc19a20dab86f2fe3afba30f11c98a7cdd41a4cb24e7d0dd2fd737"
FIXED_BASELINE_FILE = HERE / "fixed" / "cross_entropy.npz"
BASELINE_KEY = "cross_entropy"
MAX_PLAUSIBLE_ACCURACY = 97.0

# Rows are [fine, level4, level3, level2, level1] of the released CIFAR-100
# 5-level table used by the paper. Evaluator-owned copy.
PAPER_L5_ROWS = [
    [0, 4, 5, 3, 0], [1, 1, 6, 0, 0], [2, 14, 3, 0, 0], [3, 8, 3, 0, 0],
    [4, 0, 6, 0, 0], [5, 6, 0, 1, 1], [6, 7, 4, 0, 0], [7, 7, 4, 0, 0],
    [8, 18, 7, 1, 1], [9, 3, 0, 1, 1], [10, 3, 0, 1, 1], [11, 14, 3, 0, 0],
    [12, 9, 1, 1, 1], [13, 18, 7, 1, 1], [14, 7, 4, 0, 0], [15, 11, 3, 0, 0],
    [16, 3, 0, 1, 1], [17, 9, 1, 1, 1], [18, 7, 4, 0, 0], [19, 11, 3, 0, 0],
    [20, 6, 0, 1, 1], [21, 11, 3, 0, 0], [22, 5, 0, 1, 1], [23, 10, 2, 2, 1],
    [24, 7, 4, 0, 0], [25, 6, 0, 1, 1], [26, 13, 4, 0, 0], [27, 15, 4, 0, 0],
    [28, 3, 0, 1, 1], [29, 15, 4, 0, 0], [30, 0, 6, 0, 0], [31, 11, 3, 0, 0],
    [32, 1, 6, 0, 0], [33, 10, 2, 2, 1], [34, 12, 3, 0, 0], [35, 14, 3, 0, 0],
    [36, 16, 3, 0, 0], [37, 9, 1, 1, 1], [38, 11, 3, 0, 0], [39, 5, 0, 1, 1],
    [40, 5, 0, 1, 1], [41, 19, 7, 1, 1], [42, 8, 3, 0, 0], [43, 8, 3, 0, 0],
    [44, 15, 4, 0, 0], [45, 13, 4, 0, 0], [46, 14, 3, 0, 0], [47, 17, 5, 3, 0],
    [48, 18, 7, 1, 1], [49, 10, 2, 2, 1], [50, 16, 3, 0, 0], [51, 4, 5, 3, 0],
    [52, 17, 5, 3, 0], [53, 4, 5, 3, 0], [54, 2, 5, 3, 0], [55, 0, 6, 0, 0],
    [56, 17, 5, 3, 0], [57, 4, 5, 3, 0], [58, 18, 7, 1, 1], [59, 17, 5, 3, 0],
    [60, 10, 2, 2, 1], [61, 3, 0, 1, 1], [62, 2, 5, 3, 0], [63, 12, 3, 0, 0],
    [64, 12, 3, 0, 0], [65, 16, 3, 0, 0], [66, 12, 3, 0, 0], [67, 1, 6, 0, 0],
    [68, 9, 1, 1, 1], [69, 19, 7, 1, 1], [70, 2, 5, 3, 0], [71, 10, 2, 2, 1],
    [72, 0, 6, 0, 0], [73, 1, 6, 0, 0], [74, 16, 3, 0, 0], [75, 12, 3, 0, 0],
    [76, 9, 1, 1, 1], [77, 13, 4, 0, 0], [78, 15, 4, 0, 0], [79, 13, 4, 0, 0],
    [80, 16, 3, 0, 0], [81, 19, 7, 1, 1], [82, 2, 5, 3, 0], [83, 4, 5, 3, 0],
    [84, 6, 0, 1, 1], [85, 19, 7, 1, 1], [86, 5, 0, 1, 1], [87, 5, 0, 1, 1],
    [88, 8, 3, 0, 0], [89, 19, 7, 1, 1], [90, 18, 7, 1, 1], [91, 1, 6, 0, 0],
    [92, 2, 5, 3, 0], [93, 15, 4, 0, 0], [94, 6, 0, 1, 1], [95, 0, 6, 0, 0],
    [96, 17, 5, 3, 0], [97, 8, 3, 0, 0], [98, 14, 3, 0, 0], [99, 13, 4, 0, 0],
]
NUM_LEVELS = 4  # non-leaf levels 0 (root's children) .. 3 (parents of leaves)

METRIC_NAMES = (
    "accuracy", "top5_accuracy", "coarse_accuracy", "path_consistency", "fpa",
    "ms", "ahd@1", "ahd@5", "ahd@20", "hops", "hops@5", "hops@20",
)


class InvalidSubmission(Exception):
    pass


# ── evaluator-owned problem definition ───────────────────────────────────────


def load_labels() -> np.ndarray:
    raw = LABELS_FILE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != LABELS_SHA256:
        raise SystemExit(f"{LABELS_FILE} failed its integrity check; restore the evaluator files")
    labels = np.asarray(json.loads(raw)["fine_labels"], dtype=np.int64)
    if labels.shape != (NUM_EXAMPLES,) or labels.min() < 0 or labels.max() >= NUM_CLASSES:
        raise SystemExit("evaluator labels are malformed")
    return labels


class Taxonomy:
    def __init__(self) -> None:
        rows = np.asarray(PAPER_L5_ROWS, dtype=np.int64)
        if rows.shape != (NUM_CLASSES, 5) or not np.array_equal(rows[:, 0], np.arange(NUM_CLASSES)):
            raise SystemExit("evaluator taxonomy is malformed")
        # paths[c] = [level0, level1, level2, level3, fine] node values
        self.paths = np.stack([rows[:, 4], rows[:, 3], rows[:, 2], rows[:, 1], rows[:, 0]], axis=1)
        self.level_table = self.paths[:, :NUM_LEVELS]
        self.level_widths = tuple(int(self.level_table[:, lvl].max()) + 1 for lvl in range(NUM_LEVELS))
        for lvl in range(NUM_LEVELS):
            if set(self.level_table[:, lvl].tolist()) != set(range(self.level_widths[lvl])):
                raise SystemExit("evaluator taxonomy levels must use contiguous node ids")
        self.distance = self._lca_distance()
        self.max_distance = float(self.distance.max())
        self.rank, self.desired, self.eta = self._hops_tables()

    def _lca_distance(self) -> np.ndarray:
        depth = self.paths.shape[1]
        dist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                common = -1
                for d in range(depth):
                    if self.paths[i, d] != self.paths[j, d]:
                        break
                    common = d
                dist[i, j] = depth - 1 - common
        return dist

    def _hops_tables(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """rank[y, c]: preference rank of class c for true class y (0 = itself);
        desired[y]: the ideal rank vector z; eta[y]: HOPS position weights."""
        rank = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        for y in range(NUM_CLASSES):
            levels = sorted(set(self.distance[y].tolist()))
            lookup = {value: idx for idx, value in enumerate(levels)}
            rank[y] = [lookup[value] for value in self.distance[y].tolist()]
        desired = np.sort(rank, axis=1)
        eta = np.stack([eta_weights(desired[y]) for y in range(NUM_CLASSES)])
        return rank, desired, eta


def eta_weights(z: np.ndarray) -> np.ndarray:
    """Multi-step exponential-linear decay: 2^-r at the first position of rank r,
    linearly interpolated towards 2^-(r+1) over the rest of that rank's block."""
    eta = np.zeros(len(z), dtype=np.float64)
    start = 0
    while start < len(z):
        rank = int(z[start])
        end = start
        while end < len(z) and int(z[end]) == rank:
            end += 1
        count = end - start
        hi = 2.0 ** (-rank)
        lo = 2.0 ** (-(rank + 1))
        eta[start:end] = hi if count == 1 else np.linspace(hi, lo, count, endpoint=False)
        start = end
    return eta


# ── submission loading and validation ────────────────────────────────────────


def load_submission(path: Path, widths: tuple[int, ...]) -> tuple[np.ndarray, list[np.ndarray]]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            files = set(payload.files)
            leaf_key = f"{SPLIT}_leaf_scores"
            level_keys = [f"{SPLIT}_level_scores_{lvl}" for lvl in range(NUM_LEVELS)]
            missing = [key for key in [leaf_key, *level_keys] if key not in files]
            if missing:
                raise InvalidSubmission("missing required arrays: " + ", ".join(missing))
            leaf = _checked(payload[leaf_key], leaf_key, (NUM_EXAMPLES, NUM_CLASSES))
            levels = [_checked(payload[key], key, (NUM_EXAMPLES, width)) for key, width in zip(level_keys, widths)]
    except InvalidSubmission:
        raise
    except Exception as exc:  # unreadable / pickled / corrupt archive
        raise InvalidSubmission(f"unreadable prediction archive: {type(exc).__name__}: {exc}") from None
    return leaf, levels


def _checked(value: np.ndarray, key: str, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape != shape:
        raise InvalidSubmission(f"{key} has shape {arr.shape}, expected {shape}")
    if arr.dtype.kind not in "iuf":
        raise InvalidSubmission(f"{key} has non-real dtype {arr.dtype}")
    arr = arr.astype(np.float64)
    if not np.isfinite(arr).all():
        raise InvalidSubmission(f"{key} contains NaN or infinite values")
    return arr


# ── metrics (all recomputed from raw scores and evaluator-owned truth) ───────


def compute_metrics(leaf: np.ndarray, levels: list[np.ndarray], labels: np.ndarray, tax: Taxonomy) -> dict[str, float]:
    order = np.argsort(-leaf, axis=1, kind="stable")
    pred = order[:, 0]
    correct = pred == labels
    level_pred = np.stack([scores.argmax(axis=1) for scores in levels], axis=1)
    level_true = tax.level_table[labels]
    level_of_pred_leaf = tax.level_table[pred]

    dist_top = tax.distance[labels, pred]
    wrong = ~correct
    metrics = {
        "accuracy": 100.0 * float(correct.mean()),
        "top5_accuracy": 100.0 * float((order[:, :5] == labels[:, None]).any(axis=1).mean()),
        "coarse_accuracy": 100.0 * float((level_pred[:, NUM_LEVELS - 1] == level_true[:, NUM_LEVELS - 1]).mean()),
        "path_consistency": 100.0 * float((level_pred == level_of_pred_leaf).all(axis=1).mean()),
        "fpa": 100.0 * float((correct & (level_pred == level_true).all(axis=1)).mean()),
        "ms": float(dist_top[wrong].mean()) if wrong.any() else 0.0,
        "ahd@1": float(dist_top.mean()),
        "ahd@5": float(tax.distance[labels[:, None], order[:, :5]].mean()),
        "ahd@20": float(tax.distance[labels[:, None], order[:, :20]].mean()),
        "hops": hops(labels, order, None, tax),
        "hops@5": hops(labels, order, 5, tax),
        "hops@20": hops(labels, order, 20, tax),
    }
    return metrics


def hops(labels: np.ndarray, order: np.ndarray, k: int | None, tax: Taxonomy) -> float:
    """HOPS (k=None) and HOPS@k: 1 - weighted |z - z_hat| / its worst-order value."""
    z = tax.desired[labels].astype(np.float64)
    eta = tax.eta[labels]
    predicted = tax.rank[labels[:, None], order].astype(np.float64)
    if k is None:
        worst = z[:, ::-1]
        active = eta
    else:
        kk = min(k, z.shape[1])
        worst = z.copy()
        worst[:, :kk] = z[:, :kk][:, ::-1]
        active = eta.copy()
        active[:, kk:] = 0.0
    denom = (active * np.abs(z - worst)).sum(axis=1)
    num = (active * np.abs(z - predicted)).sum(axis=1)
    safe = np.where(denom > 0, denom, 1.0)
    score = np.where(denom > 0, 1.0 - num / safe, (predicted[:, 0] == 0).astype(np.float64))
    if k is not None:
        score = np.maximum(0.0, score)
    return float(score.mean())


def failed_metrics(tax: Taxonomy) -> dict[str, float]:
    worst = {name: 0.0 for name in METRIC_NAMES}
    for name in ("ms", "ahd@1", "ahd@5", "ahd@20"):
        worst[name] = tax.max_distance
    return worst


def score_file(path: Path, labels: np.ndarray, tax: Taxonomy) -> tuple[dict[str, float], dict[str, object]]:
    meta: dict[str, object] = {"file": str(path), "num_examples": NUM_EXAMPLES, "split": SPLIT}
    try:
        leaf, levels = load_submission(path, tax.level_widths)
        metrics = compute_metrics(leaf, levels, labels, tax)
        if metrics["accuracy"] >= MAX_PLAUSIBLE_ACCURACY:
            raise InvalidSubmission(
                f"fine accuracy {metrics['accuracy']:.2f}% is at or above the plausibility ceiling "
                f"{MAX_PLAUSIBLE_ACCURACY}% for CIFAR-100 (consistent with test-label lookup)"
            )
    except InvalidSubmission as exc:
        meta.update({"valid": False, "reason": str(exc)})
        return failed_metrics(tax), meta
    meta["valid"] = True
    return metrics, meta


def evaluate_all(prediction_dir: Path, output_path: Path) -> dict[str, object]:
    labels = load_labels()
    tax = Taxonomy()
    if not FIXED_BASELINE_FILE.exists():
        raise SystemExit(f"missing evaluator-owned baseline {FIXED_BASELINE_FILE}")
    results: dict[str, dict[str, float]] = {}
    methods_meta: dict[str, dict[str, object]] = {}
    baseline_metrics, baseline_meta = score_file(FIXED_BASELINE_FILE, labels, tax)
    if not baseline_meta["valid"]:
        raise SystemExit(f"evaluator-owned baseline is invalid: {baseline_meta['reason']}")
    results[BASELINE_KEY] = baseline_metrics
    methods_meta[BASELINE_KEY] = {**baseline_meta, "source": "evaluator-owned frozen baseline"}
    ignored: list[str] = []
    for path in sorted(prediction_dir.glob("*.npz")) if prediction_dir.is_dir() else []:
        key = path.stem
        if key == BASELINE_KEY:
            ignored.append(f"{path}: '{BASELINE_KEY}' is reserved for the evaluator-owned baseline")
            continue
        results[key], methods_meta[key] = score_file(path, labels, tax)

    payload = {
        "experiments": {
            EXPERIMENT: {
                "results": results,
                "_meta": {
                    "split": SPLIT,
                    "num_examples": NUM_EXAMPLES,
                    "hierarchy": "paper_l5",
                    "prediction_dir": str(prediction_dir),
                    "plausibility_ceiling_accuracy": MAX_PLAUSIBLE_ACCURACY,
                    "methods": methods_meta,
                    "ignored_files": ignored,
                },
            }
        }
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", default="outputs/predictions")
    parser.add_argument("--output", default="scoring/scores_test.json")
    args = parser.parse_args()
    payload = evaluate_all(Path(args.prediction_dir), Path(args.output))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
