"""Collect measured metrics without inventing an unspecified scalar rank."""
from __future__ import annotations
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
EXP = 'cifar100_hierarchical_classification'


def main():
    summary = {}
    for path in sorted(ROOT.glob('*/result.json')):
        folder = path.parent
        row = json.loads(path.read_text())
        score_path = folder / 'scores_test.json'
        if score_path.exists() and row['test_returncode'] == 0:
            scores = json.loads(score_path.read_text())['experiments'][EXP]
            row['metrics'] = scores['results'].get('hier_cos')
            row['baseline_metrics'] = scores['results'].get('cross_entropy')
            row['test_examples'] = scores['_meta']['methods'].get('hier_cos',{}).get('num_examples')
        summary[folder.name] = row
    ref = summary.get('reference',{}).get('metrics',{})
    for name,row in summary.items():
        metrics = row.get('metrics')
        if metrics and ref:
            row['delta_hops'] = metrics['hops'] - ref['hops']
            row['delta_accuracy'] = metrics['accuracy'] - ref['accuracy']
            row['delta_fpa'] = metrics['fpa'] - ref['fpa']
            if name.startswith('shortcut_'):
                row['primary_score_at_least_reference'] = metrics['hops'] >= ref['hops'] - 1e-12
    seeds = [summary.get(f'seed_{seed}',{}).get('metrics',{}).get('hops') for seed in [7,8,9]]
    if all(value is not None for value in seeds):
        summary['seed_spread'] = {'seeds':[7,8,9], 'hops_values':seeds,
            'range':max(seeds)-min(seeds), 'sample_standard_deviation':statistics.stdev(seeds)}
        summary['seed_metric_spreads'] = {}
        for metric in ref:
            values = [summary[f'seed_{seed}']['metrics'][metric] for seed in [7,8,9]]
            summary['seed_metric_spreads'][metric] = {'values':values,'range':max(values)-min(values)}
    (ROOT / 'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True))
    for name,row in summary.items():
        if isinstance(row,dict) and row.get('metrics'):
            metrics = row['metrics']
            print(f"{name}: HOPS={metrics['hops']:.9f}, accuracy={metrics['accuracy']:.3f}, FPA={metrics['fpa']:.3f}, delta={row.get('delta_hops',0):+.9f}")


if __name__ == '__main__':
    main()
