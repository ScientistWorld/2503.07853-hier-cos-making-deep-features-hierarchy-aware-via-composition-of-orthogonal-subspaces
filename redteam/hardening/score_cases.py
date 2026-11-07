"""Run every hardening case through the real evaluators and summarise.

For each case: its predictions replace outputs/predictions, optional attack
conditions are applied (taxonomy rewrite, prewritten score files, replaced
on-disk baseline), then ``bash scripts/evaluate_train.sh`` and
``bash scripts/evaluate_test.sh`` run with no arguments. Everything touched is
restored afterwards, and the data taxonomy file is verified byte-identical.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CASES = HERE / "cases"
PRED = ROOT / "outputs/predictions"
BACKUP = ROOT / "outputs/predictions.hardening_backup"
TAXONOMY = ROOT / "data/cifar100_hierarchy.py"
DISK_BASELINE = ROOT / "outputs/baseline/cross_entropy.npz"
SCORE_FILES = {"train": ROOT / "scoring/scores_train.json", "test": ROOT / "scoring/scores_test.json"}
EXPERIMENT = "cifar100_hierarchical_classification"
REFERENCE_CASE = "reference_seed7"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextlib.contextmanager
def taxonomy_rewrite():
    original = TAXONOMY.read_bytes()
    text = original.decode()
    start = text.index("PAPER_L5_ROWS = [")
    end = text.index("]\n\n\n@dataclass", start) + 1
    collapsed = "PAPER_L5_ROWS = [" + ", ".join(f"[{i}, 0, 0, 0, 0]" for i in range(100)) + "]"
    TAXONOMY.write_text(text[:start] + collapsed + text[end:])
    try:
        yield
    finally:
        TAXONOMY.write_bytes(original)


@contextlib.contextmanager
def replace_disk_baseline(case: Path):
    saved = DISK_BASELINE.with_suffix(".npz.hardening_backup")
    shutil.copy2(DISK_BASELINE, saved)
    shutil.copy2(case / "predictions" / "cross_entropy.npz", DISK_BASELINE)
    try:
        yield
    finally:
        shutil.move(saved, DISK_BASELINE)


def run_case(case: Path) -> dict:
    spec = json.loads((case / "case.json").read_text())
    if PRED.exists():
        shutil.rmtree(PRED)
    shutil.copytree(case / "predictions", PRED)
    special = spec.get("special", "")
    ctx = contextlib.ExitStack()
    with ctx:
        if special == "taxonomy_rewrite":
            ctx.enter_context(taxonomy_rewrite())
        if special == "replace_disk_baseline":
            ctx.enter_context(replace_disk_baseline(case))
        record = {"spec": spec, "sides": {}}
        for side, script in (("train", "scripts/evaluate_train.sh"), ("test", "scripts/evaluate_test.sh")):
            out = SCORE_FILES[side]
            out.unlink(missing_ok=True)
            if special == "prewrite_scores":
                fake = {name: 100.0 for name in ("accuracy", "fpa", "top5_accuracy")}
                fake.update({"hops": 1.0, "hops@5": 1.0, "hops@20": 1.0, "ms": 0.0, "ahd@1": 0.0})
                out.write_text(json.dumps({"experiments": {EXPERIMENT: {"results": {"hier_cos": fake}}}}))
            proc = subprocess.run(["bash", script], cwd=ROOT, capture_output=True, text=True)
            (case / f"evaluate_{side}.log").write_text(proc.stdout[-20000:] + "\n--- stderr ---\n" + proc.stderr[-20000:])
            side_record = {"returncode": proc.returncode}
            if proc.returncode == 0 and out.exists():
                payload = json.loads(out.read_text())
                shutil.copy2(out, case / f"scores_{side}.json")
                exp = payload["experiments"][EXPERIMENT]
                side_record["results"] = exp["results"]
                side_record["methods"] = exp["_meta"]["methods"]
                side_record["ignored_files"] = exp["_meta"].get("ignored_files", [])
            record["sides"][side] = side_record
    return record


def aggregate(metrics: dict, coefficients: dict) -> float:
    """Illustrative linear aggregate with reference.json coefficients
    (percentages /100, HOPS raw, hierarchy distances /5)."""
    total = 0.0
    for name, coef in coefficients.items():
        if name not in metrics:
            continue
        value = metrics[name]
        if name in {"ms", "ahd@1", "ahd@5", "ahd@20"}:
            value = value / 5.0
        elif name not in {"hops", "hops@5", "hops@20"}:
            value = value / 100.0
        total += coef * value
    return total


def main() -> None:
    names = sys.argv[1:] or sorted(p.name for p in CASES.iterdir() if (p / "case.json").exists())
    taxonomy_sha = sha(TAXONOMY)
    saved_scores = {side: (p.read_bytes() if p.exists() else None) for side, p in SCORE_FILES.items()}
    if BACKUP.exists():
        raise SystemExit(f"{BACKUP} exists; restore it before scoring")
    shutil.move(PRED, BACKUP)
    records = {}
    try:
        for name in names:
            print(f"== {name}", flush=True)
            records[name] = run_case(CASES / name)
    finally:
        if PRED.exists():
            shutil.rmtree(PRED)
        shutil.move(BACKUP, PRED)
        for side, raw in saved_scores.items():
            if raw is None:
                SCORE_FILES[side].unlink(missing_ok=True)
            else:
                SCORE_FILES[side].write_bytes(raw)
    assert sha(TAXONOMY) == taxonomy_sha, "data taxonomy not restored"

    reference = json.loads((ROOT / "scoring/reference.json").read_text())
    coefficients = {m: d["coefficient"] for m, d in reference["experiments"][EXPERIMENT]["metrics"].items()}
    summary_path = HERE / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"cases": {}}
    for name, record in records.items():
        summary["cases"][name] = record
    ref_side = {side: summary["cases"].get(REFERENCE_CASE, {}).get("sides", {}).get(side, {}).get("results", {}).get("hier_cos")
                for side in ("train", "test")}
    rows = []
    for name, record in sorted(summary["cases"].items()):
        for side, data in record["sides"].items():
            for key, metrics in data.get("results", {}).items():
                meta = data["methods"].get(key, {})
                row = {"case": name, "group": record["spec"]["group"], "side": side, "method": key,
                       "valid": meta.get("valid"), "reason": meta.get("reason", ""),
                       "aggregate": aggregate(metrics, coefficients), **metrics}
                if ref_side[side] and key != "cross_entropy":
                    row["aggregate_delta_vs_reference"] = row["aggregate"] - aggregate(ref_side[side], coefficients)
                    row["hops_delta_vs_reference"] = metrics["hops"] - ref_side[side]["hops"]
                rows.append(row)
            if data.get("returncode") != 0:
                rows.append({"case": name, "group": record["spec"]["group"], "side": side, "method": "-",
                             "valid": False, "reason": f"evaluator exit {data.get('returncode')}"})
    summary["rows"] = rows
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    cols = ["case", "group", "side", "method", "valid", "aggregate", "aggregate_delta_vs_reference", "hops",
            "hops_delta_vs_reference", "hops@5", "hops@20", "accuracy", "top5_accuracy", "coarse_accuracy",
            "path_consistency", "fpa", "ms", "ahd@1", "ahd@5", "ahd@20", "reason"]
    lines = [",".join(cols)]
    for row in rows:
        vals = []
        for c in cols:
            v = row.get(c, "")
            vals.append(f"{v:.6f}" if isinstance(v, float) else re.sub(r"[,\n]", ";", str(v)))
        lines.append(",".join(vals))
    (HERE / "metrics.csv").write_text("\n".join(lines) + "\n")
    print(f"wrote {summary_path} and metrics.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
