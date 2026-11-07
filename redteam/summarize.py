"""Summarize completed audit runs without running models or evaluators."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "redteam"
EXPERIMENT = "cifar100_hierarchical_classification"


def main():
    cases = json.loads((AUDIT / "cases.json").read_text())
    reference_spec = json.loads((ROOT / "scoring/reference.json").read_text())
    metrics = reference_spec["experiments"][EXPERIMENT]["metrics"]
    runs = {}
    for case in cases:
        directory = AUDIT / case
        result = {"status": json.loads((directory / "status.json").read_text())}
        if (directory / "scores_test.json").exists():
            experiment = json.loads((directory / "scores_test.json").read_text())["experiments"][EXPERIMENT]
            result.update(results=experiment["results"], meta=experiment["_meta"])
        runs[case] = result
    reference = runs["reference_artifacts"]["results"]["hier_cos"]
    seeds = [runs[f"reference_seed_{seed}"]["results"]["hier_cos"] for seed in (7, 8, 9)]
    seed_stats = {}
    for metric in metrics:
        values = [seed[metric] for seed in seeds]
        seed_stats[metric] = dict(min=min(values), max=max(values), spread=max(values)-min(values),
                                  mean=statistics.mean(values), sample_std=statistics.stdev(values))
    for case, run in runs.items():
        if "results" not in run:
            continue
        scores = run["results"]["hier_cos"]
        signed = {m: (scores[m] - reference[m]) * (1 if spec["higher_is_better"] else -1)
                  for m, spec in metrics.items()}
        run["delta_vs_reference"] = {m: scores[m]-reference[m] for m in metrics}
        run["all_metrics_tie"] = all(abs(v) < 1e-10 for v in signed.values())
        run["dominates_reference"] = all(v >= -1e-10 for v in signed.values()) and any(v > 1e-10 for v in signed.values())
    summary = dict(primary_metric="hops", aggregate_rank_formula_available=False,
                   reference_case="reference_artifacts", seed_cases=[f"reference_seed_{s}" for s in (7, 8, 9)],
                   seed_stats=seed_stats, runs=runs)
    (AUDIT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (AUDIT / "metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["case", "test_exit_code", "num_examples", *metrics])
        for case, run in runs.items():
            if "results" in run:
                writer.writerow([case, run["status"]["test"], run["meta"]["methods"]["hier_cos"]["num_examples"],
                                 *[run["results"]["hier_cos"][m] for m in metrics]])
            else:
                writer.writerow([case, run["status"]["test"], "", *[""]*len(metrics)])
    print(f"Summarized {len(runs)} runs; HOPS seed spread {seed_stats['hops']['spread']:.9f}")


if __name__ == "__main__":
    main()
