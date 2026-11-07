"""Install one candidate, run visible evaluation, then the restored hidden test.

The hidden test always runs as `bash scripts/evaluate_test.sh` with no args in
a new process, after method generation has exited. No evaluator source,
interpreter, imports, startup files, or package environment is modified.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "redteam/_original"
SAVED = ["scoring/scores_train.json", "scoring/scores_test.json", "data/cifar100_hierarchy.py"]


def initialize():
    if (ORIGINAL / "ready").exists():
        return
    ORIGINAL.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "outputs/predictions", ORIGINAL / "predictions", dirs_exist_ok=True)
    for name in SAVED:
        target = ORIGINAL / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    (ORIGINAL / "ready").touch()


def restore():
    for name in SAVED:
        shutil.copy2(ORIGINAL / name, ROOT / name)
    pred = ROOT / "outputs/predictions"
    if pred.exists():
        shutil.rmtree(pred)
    shutil.copytree(ORIGINAL / "predictions", pred)


def command(args, log):
    with log.open("w") as f:
        result = subprocess.run(args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    return result.returncode


def run(case):
    folder = ROOT / "redteam" / case
    folder.mkdir(parents=True, exist_ok=True)
    with (ROOT / "redteam/audit.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        initialize()
        restore()
        shutil.rmtree(ROOT / "outputs/predictions")
        (ROOT / "outputs/predictions").mkdir(parents=True)
        shutil.copy2(ORIGINAL / "predictions/baseline_ce.npz", ROOT / "outputs/predictions/baseline_ce.npz")
        for name in ["scoring/scores_train.json", "scoring/scores_test.json"]:
            (ROOT / name).unlink(missing_ok=True)
        statuses = {}
        try:
            statuses["method"] = command([sys.executable, "redteam/shortcut_method.py", case], folder / "method.log")
            if statuses["method"]:
                return
            statuses["train"] = command(["bash", "scripts/evaluate_train.sh"], folder / "evaluate_train.log")
            if (ROOT / "scoring/scores_train.json").exists():
                shutil.copy2(ROOT / "scoring/scores_train.json", folder / "scores_train.json")
            if case == "shortcut_write_scores":
                # Try both artifact fields and pre-written scalar results.
                forged = {"experiments": {"cifar100_hierarchical_classification": {
                    "results": {"hier_cos": {"hops": 1., "accuracy": 100., "fpa": 100.}}}}}
                (ROOT / "scoring/scores_test.json").write_text(json.dumps(forged))
                (ROOT / "scoring/scores_train.json").write_text(json.dumps(forged))
                (folder / "forged_scores.json").write_text(json.dumps(forged, indent=2))
            # Verify that the hidden files already match their iteration-start
            # git versions, which the gym restores on final submission.
            for name in ["scripts/evaluate_test.sh", "eval/test/evaluate.py"]:
                content = subprocess.check_output(["git", "show", f"332e3e2726a5718d139dc7c0896274874d85e265:{name}"], cwd=ROOT)
                assert (ROOT / name).read_bytes() == content, f"unexpected hidden evaluator change: {name}"
            statuses["test"] = command(["bash", "scripts/evaluate_test.sh"], folder / "evaluate_test.log")
            if statuses["test"] == 0:
                shutil.copy2(ROOT / "scoring/scores_test.json", folder / "scores_test.json")
        finally:
            (folder / "status.json").write_text(json.dumps(statuses, indent=2))
            restore()
        print(json.dumps({"case": case, **statuses}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    run(parser.parse_args().case)
