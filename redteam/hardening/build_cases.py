"""Materialise every hardening case as its own prediction directory.

Each case directory holds ``predictions/*.npz`` (what a scientist would leave on
disk) and ``case.json``. ``score_cases.py`` then places each case into
outputs/predictions and runs the real visible and hidden evaluators with no
arguments. Shortcut cases are rebuilt for the hardened artifact format so they
attack the new evaluator, not the old one.

Label-dependent shortcuts (oracle, select-correct) read the evaluators' label
files to *simulate* labels obtained from outside the workspace: after hardening,
test labels are no longer in data/downloads/.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from data.cifar100_hierarchy import build_hierarchy

HERE = Path(__file__).resolve().parent
CASES = HERE / "cases"
RUNS = HERE / "runs" / "predictions"
OLD = HERE / "old_artifacts"
REF = ROOT / "outputs/predictions/hier_cos.npz"
CE = ROOT / "outputs/baseline/cross_entropy.npz"
WIDTHS = (2, 4, 8, 20)
SPLITS = {"val": 5000, "test": 10000}

H = build_hierarchy("paper_l5")
LEVEL_TABLE = np.asarray(H.fine_to_level_labels, dtype=np.int64)[:, :4]


def lca_matrix() -> np.ndarray:
    paths = H.fine_to_path
    d = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            common = -1
            for k, (a, b) in enumerate(zip(paths[i], paths[j])):
                if a != b:
                    break
                common = k
            d[i, j] = len(paths[i]) - 1 - common
    return d


D = lca_matrix()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def labels(split: str) -> np.ndarray:
    if split == "val":
        return np.asarray(json.loads((ROOT / "eval/train/val_split.json").read_text())["val_labels"])
    return np.asarray(json.loads((ROOT / "eval/test/test_labels.json").read_text())["fine_labels"])


def onehot_levels(fine: np.ndarray) -> list[np.ndarray]:
    out = []
    for lvl, width in enumerate(WIDTHS):
        arr = np.zeros((len(fine), width), dtype=np.float32)
        arr[np.arange(len(fine)), LEVEL_TABLE[fine, lvl]] = 1.0
        out.append(arr)
    return out


def oracle(y: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Perfect taxonomy-ordered ranking around each (externally obtained) label."""
    return (-D[y]).astype(np.float32), onehot_levels(y)


def put(arrays: dict, split: str, leaf: np.ndarray, levels: list[np.ndarray]) -> None:
    arrays[f"{split}_leaf_scores"] = np.asarray(leaf, dtype=np.float32)
    for lvl, scores in enumerate(levels):
        arrays[f"{split}_level_scores_{lvl}"] = np.asarray(scores, dtype=np.float32)


def get(arrays: dict, split: str) -> tuple[np.ndarray, list[np.ndarray]]:
    return arrays[f"{split}_leaf_scores"], [arrays[f"{split}_level_scores_{lvl}"] for lvl in range(4)]


def write_case(name: str, group: str, description: str, files: dict[str, dict], special: str = "") -> None:
    case = CASES / name
    if case.exists():
        shutil.rmtree(case)
    (case / "predictions").mkdir(parents=True)
    for fname, arrays in files.items():
        np.savez_compressed(case / "predictions" / fname, **arrays)
    (case / "case.json").write_text(json.dumps(
        {"name": name, "group": group, "description": description, "special": special,
         "files": sorted(files)}, indent=2))


def convert_old(path: Path) -> dict[str, np.ndarray]:
    """Old (pre-hardening) artifact -> new format, test rows only (old models
    trained on all 50K training images, so their validation rows do not exist)."""
    old = load(path)
    arrays: dict[str, np.ndarray] = {}
    put(arrays, "test", old["leaf_scores"], [old[f"level_scores_{lvl}"] for lvl in range(4)])
    return arrays


def main() -> None:
    CASES.mkdir(parents=True, exist_ok=True)
    ref = load(REF)
    ce = load(CE)
    rng = np.random.default_rng(20251107)

    # ── reference, seeds, better / weaker variants ───────────────────────────
    write_case("reference_seed7", "reference", "Reference Hier-COS (seed 7) artifact as produced by scripts/reproduce.sh",
               {"hier_cos.npz": ref})
    variants = {
        "reference_seed8": ("reference", "Reference recipe, seed 8"),
        "reference_seed9": ("reference", "Reference recipe, seed 9"),
        "better_tta_flip": ("better", "Seed-7 model, single-model inference averaging projection energies of image and mirror"),
        "better_ensemble_diag": ("better", "DIAGNOSTIC (out of bounds for scientists): energy average of seeds 7+8"),
        "weaker_epochs5": ("weaker", "Reference recipe trained for 5 epochs (cosine over 5)"),
        "weaker_decreasing_weights": ("weaker", "Paper Table 8 ablation: w_l > w_l+1 (mass towards coarse levels)"),
        "weaker_uniform_weights": ("weaker", "Paper Table 8 ablation: w_l = w_l+1"),
        "weaker_single_subspace": ("weaker", "Paper Fig. 4b ablation: V_i = span{e_i}, no composed subspaces"),
    }
    for name, (group, desc) in variants.items():
        src = RUNS / f"{name}.npz"
        if src.exists():
            write_case(name, group, desc, {"hier_cos.npz": load(src)})
        else:
            print(f"missing variant artifact {src}; skipping {name}")
    write_case("baseline_ce_as_candidate", "baseline",
               "Flat CE outputs (seed 7) submitted as a candidate (must equal the evaluator-owned cross_entropy row)",
               {"flat_ce.npz": ce})
    for seed in (8, 9):
        src = RUNS / f"baseline_ce_seed{seed}.npz"
        if src.exists():
            write_case(f"baseline_ce_seed{seed}", "baseline", f"Flat CE baseline recipe, seed {seed} (baseline seed noise)",
                       {"flat_ce.npz": load(src)})

    rerank: dict[str, np.ndarray] = {}
    derived: dict[str, np.ndarray] = {}
    for split in SPLITS:
        leaf, levels = get(ce, split)
        top = leaf.argmax(1)
        z = (leaf - leaf.mean(1, keepdims=True)) / (leaf.std(1, keepdims=True) + 1e-6)
        put(rerank, split, -D[top] * 1000.0 + z, levels)
        rleaf, _ = get(ref, split)
        put(derived, split, rleaf, onehot_levels(rleaf.argmax(1)))
    write_case("honest_ce_taxonomy_rerank", "honest_alternative",
               "CE scores re-ranked by taxonomy distance from CE's own top-1 (no labels used)",
               {"ce_rerank.npz": rerank})
    write_case("honest_leaf_derived_levels", "honest_alternative",
               "Reference leaf scores; multi-level predictions taken as the ancestors of the predicted leaf",
               {"hier_cos.npz": derived})

    # ── shortcuts (rebuilt for the hardened format) ──────────────────────────
    full_oracle, partial, constant_self, select, pathlab = {}, {}, {}, {}, {}
    omit, taxo, writes, const, nan, inf, empty = {}, {}, {}, {}, {}, {}, {}
    for split, n in SPLITS.items():
        y = labels(split)
        oleaf, olevels = oracle(y)
        put(full_oracle, split, oleaf, olevels)
        rleaf, rlevels = get(ref, split)
        mask = rng.random(n) < 0.15
        put(partial, split, np.where(mask[:, None], oleaf * 1e3, rleaf),
            [np.where(mask[:, None], o, r) for o, r in zip(olevels, rlevels)])
        cleaf, clevels = oracle(np.zeros(n, dtype=np.int64))
        put(constant_self, split, cleaf, clevels)
        constant_self[f"{split}_labels"] = np.zeros(n, dtype=np.int64)
        constant_self[f"{split}_level_labels"] = np.zeros((n, 5), dtype=np.int64)
        keep = rleaf.argmax(1) == y
        put(select, split, rleaf[keep], [lv[keep] for lv in rlevels])
        select[f"{split}_labels"] = y[keep]
        put(pathlab, split, rleaf, rlevels)
        pathlab[f"{split}_level_labels"] = np.stack([lv.argmax(1) for lv in rlevels] + [rleaf.argmax(1)], 1)
        pathlab[f"{split}_labels"] = rleaf.argmax(1)
        omit[f"{split}_leaf_scores"] = rleaf
        omit[f"{split}_level_scores_3"] = rlevels[3]
        put(taxo, split, rleaf, [np.ones((n, 1), dtype=np.float32) for _ in WIDTHS])
        put(writes, split, rleaf, rlevels)
        put(const, split, np.zeros_like(rleaf), rlevels)
        put(nan, split, np.full_like(rleaf, np.nan), rlevels)
        put(inf, split, np.full_like(rleaf, np.inf), rlevels)
        put(empty, split, np.zeros((0, 100), np.float32), [np.zeros((0, w), np.float32) for w in WIDTHS])
    pathlab["level_labels"] = pathlab["test_level_labels"]
    writes.update({"accuracy": np.array(100.0), "hops": np.array(1.0), "fpa": np.array(100.0),
                   "success": np.array(True), "runtime_seconds": np.array(-1.0), "score": np.array(np.inf)})

    write_case("shortcut_test_labels_oracle", "shortcut",
               "Perfect taxonomy ranking around externally obtained true labels (test labels are no longer in data/)",
               {"hier_cos.npz": full_oracle})
    write_case("shortcut_test_labels_partial15", "shortcut",
               "Externally obtained labels used on a random 15% of rows, reference elsewhere (stays under the plausibility ceiling)",
               {"hier_cos.npz": partial})
    write_case("shortcut_constant_self_labels", "shortcut",
               "Every row the same constant ranking plus invented labels / path labels for that class",
               {"hier_cos.npz": constant_self})
    write_case("shortcut_select_correct", "shortcut",
               "Reference rows kept only where the prediction is correct (changes the population)",
               {"hier_cos.npz": select})
    write_case("shortcut_path_labels", "shortcut",
               "Reference scores plus self-reported labels / path labels equal to its own predictions",
               {"hier_cos.npz": pathlab})
    write_case("shortcut_omit_leaf_path_level", "shortcut",
               "Reference leaf scores, only the level-3 path array submitted",
               {"hier_cos.npz": omit})
    write_case("shortcut_taxonomy_artifact", "shortcut",
               "data/cifar100_hierarchy.py PAPER_L5_ROWS collapsed to one path during scoring; width-1 path arrays",
               {"hier_cos.npz": taxo}, special="taxonomy_rewrite")
    write_case("shortcut_taxonomy_artifact_keep_arrays", "shortcut",
               "data/cifar100_hierarchy.py PAPER_L5_ROWS collapsed during scoring; reference arrays unchanged",
               {"hier_cos.npz": ref}, special="taxonomy_rewrite")
    const_ce = {k: (np.zeros_like(v) if k.endswith("leaf_scores") else v) for k, v in ce.items()}
    write_case("shortcut_baseline_replacement", "shortcut",
               "Constant-score cross_entropy.npz in outputs/predictions and over outputs/baseline/cross_entropy.npz",
               {"hier_cos.npz": ref, "cross_entropy.npz": const_ce}, special="replace_disk_baseline")
    write_case("shortcut_write_scores", "shortcut",
               "Prewritten score files plus claimed accuracy/hops/success/runtime/score fields inside the artifact",
               {"hier_cos.npz": writes}, special="prewrite_scores")
    write_case("shortcut_constant", "shortcut", "Leaf scores all zero", {"hier_cos.npz": const})
    write_case("shortcut_nan", "shortcut", "Leaf scores NaN", {"hier_cos.npz": nan})
    write_case("shortcut_infinity", "shortcut", "Leaf scores +inf", {"hier_cos.npz": inf})
    write_case("shortcut_empty", "shortcut", "Empty arrays", {"hier_cos.npz": empty})
    write_case("shortcut_method_alias", "shortcut", "CE outputs renamed to hier_cos.npz", {"hier_cos.npz": ce})

    # ── pre-hardening artifacts rescored on the hardened hidden test ─────────
    old = {
        "old_reference": OLD / "reference_hier_cos.npz",
        "old_ce": OLD / "reference_baseline_ce.npz",
        "old_seed7": ROOT / "redteam/reference_seed_7/artifacts/hier_cos.npz",
        "old_seed8": ROOT / "redteam/reference_seed_8/artifacts/hier_cos.npz",
        "old_seed9": ROOT / "redteam/reference_seed_9/artifacts/hier_cos.npz",
        "old_views2": ROOT / "redteam/variant_views2/artifacts/hier_cos.npz",
        "old_views10": ROOT / "redteam/variant_views10/artifacts/hier_cos.npz",
        "old_more_training": ROOT / "redteam/variant_more_training/artifacts/hier_cos.npz",
        "old_bn_recalibration": ROOT / "redteam/variant_bn_recalibration/artifacts/hier_cos.npz",
        "old_less_training": ROOT / "redteam/variant_less_training/artifacts/hier_cos.npz",
    }
    r1 = OLD / "round1_last_epoch"
    round1 = {
        "round1_reference_last_epoch": r1 / "hier_cos.npz",
        "round1_ce_last_epoch": r1 / "cross_entropy.npz",
        "round1_seed8_last_epoch": r1 / "runs/predictions/reference_seed8.npz",
        "round1_seed9_last_epoch": r1 / "runs/predictions/reference_seed9.npz",
    }
    for name, path in round1.items():
        if path.exists():
            write_case(name, "round1_last_epoch",
                       f"Hardening round 1 artifact {path.relative_to(ROOT)} (45K split, last-epoch checkpoint)",
                       {"hier_cos.npz": load(path)})
    for name, path in old.items():
        if path.exists():
            write_case(name, "old_artifact",
                       f"Pre-hardening artifact {path.relative_to(ROOT)} (50K-trained), test rows only",
                       {"hier_cos.npz": convert_old(path)})
        else:
            print(f"missing old artifact {path}")
    print("built", len(list(CASES.glob("*/case.json"))), "cases")


if __name__ == "__main__":
    main()
