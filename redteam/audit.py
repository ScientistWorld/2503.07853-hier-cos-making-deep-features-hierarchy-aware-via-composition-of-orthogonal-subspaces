"""Install one real artifact-producing variant, run train and hidden test, restore.

Shortcut construction imports only modules available to a future agent. Hidden
evaluation runs via its unmodified entry point, without arguments, in a new
process. This harness never changes evaluator code or Python configuration.
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from data.cifar100_hierarchy import build_hierarchy
from eval.train.evaluate import hops, lca_distance_matrix

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'redteam'
PRED = ROOT / 'outputs/predictions'
EXP = 'cifar100_hierarchical_classification'


def initialize_artifacts():
    artifacts = AUDIT / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    for name in ['hier_cos', 'baseline_ce']:
        destination = artifacts / f'{name}.npz'
        if not destination.exists():
            if not (PRED / f'{name}.npz').exists():
                raise FileNotFoundError('Generate the supplied reference/baseline predictions first')
            shutil.copy2(PRED / f'{name}.npz', destination)


def load(name='hier_cos'):
    with np.load(AUDIT / 'artifacts' / f'{name}.npz', allow_pickle=False) as source:
        return {key: np.array(source[key]) for key in source.files}


def write(obj, name='hier_cos'):
    np.savez_compressed(PRED / f'{name}.npz', **obj)


def set_levels(obj, hierarchy_name='paper_l5', perfect=False):
    hierarchy = build_hierarchy(hierarchy_name)
    labels = obj['labels'].astype(int)
    for key in list(obj):
        if key.startswith('level_scores_'):
            del obj[key]
    obj['hierarchy'] = np.array(hierarchy_name)
    obj['level_labels'] = np.asarray(hierarchy.fine_to_level_labels)[labels]
    if perfect:
        for level, nodes in enumerate(hierarchy.level_node_indices):
            obj[f'level_scores_{level}'] = np.eye(len(nodes))[obj['level_labels'][:,level]]
    else:
        logits = obj['leaf_scores']
        exp = np.exp(logits-logits.max(axis=1, keepdims=True))
        prob = exp / exp.sum(axis=1, keepdims=True)
        for level, leaf_sets in enumerate(hierarchy.level_to_leaf_sets):
            obj[f'level_scores_{level}'] = np.stack([prob[:,s].sum(axis=1) for s in leaf_sets], axis=1)
    obj['coarse_scores'] = obj['level_scores_0']


def prepare(name):
    obj = load()
    details = {'case':name, 'uses_hidden_code_or_answers':False}
    if name in ['reference', 'shortcut_forged_metrics', 'shortcut_replace_baseline']:
        pass
    elif name.startswith('seed_') or name.startswith('variant_'):
        path = AUDIT / name / 'prediction.npz'
        if not path.exists():
            seed = int(name.split('_')[1]) if name.startswith('seed_') else 7
            command = ['python', '-m', 'redteam.train_variants', '--seed',str(seed),'--gpu','0']
            if name == 'variant_longer_training':
                command.append('--extend')
            subprocess.run(command, cwd=ROOT, check=True)
        shutil.copy2(path, PRED / 'hier_cos.npz')
        return details
    elif name == 'shortcut_visible_test_labels':
        with (ROOT / 'data/downloads/cifar-100-python/test').open('rb') as f:
            dataset = pickle.load(f, encoding='latin1')
        obj['labels'] = np.asarray(dataset['fine_labels'])
        obj['leaf_scores'] = -lca_distance_matrix(build_hierarchy('paper_l5'))[obj['labels']]
        set_levels(obj, perfect=True)
        details['answer_source'] = 'data/downloads/cifar-100-python/test: fine_labels'
    elif name == 'shortcut_forged_ground_truth':
        obj['labels'] = obj['leaf_scores'].argmax(axis=1)
        obj['level_labels'] = np.stack([obj[f'level_scores_{i}'].argmax(axis=1) for i in range(5)], axis=1)
    elif name == 'shortcut_forged_path_labels':
        obj['level_labels'] = np.stack([obj[f'level_scores_{i}'].argmax(axis=1) for i in range(5)], axis=1)
    elif name == 'shortcut_cherry_pick_test_size':
        labels = obj['labels']
        order = np.argsort(-obj['leaf_scores'], axis=1)
        distances = lca_distance_matrix(build_hierarchy('paper_l5'))
        path_correct = np.all(np.stack([obj[f'level_scores_{i}'].argmax(axis=1) for i in range(5)],axis=1)
            == obj['level_labels'], axis=1)
        candidates = np.flatnonzero((order[:,0] == labels) & path_correct)
        # Only the visible evaluator is used to select the best 100 submitted
        # examples. The hidden evaluator is allowed to resplit this reduced set.
        candidate_scores = np.asarray([hops(labels[i:i+1],order[i:i+1],None,distances) for i in candidates])
        keep = candidates[np.argsort(-candidate_scores)[:100]]
        details.update({'original_examples':len(labels), 'submitted_examples':len(keep)})
        obj = {key:(value[keep] if value.ndim and value.shape[0] == len(labels) else value)
            for key,value in obj.items()}
    elif name == 'shortcut_collapse_fixed_taxonomy':
        path = ROOT / 'data/cifar100_hierarchy.py'
        path.write_text(path.read_text() + '\n\n# Shortcut: replace the supplied fixed taxonomy distance artifact.\n'
            '_shortcut_original_build_hierarchy = build_hierarchy\n'
            'def build_hierarchy(name="public"):\n'
            '    from dataclasses import replace\n'
            '    hierarchy = _shortcut_original_build_hierarchy(name)\n'
            '    return replace(hierarchy, fine_to_path=[hierarchy.fine_to_path[0].copy() for _ in range(100)])\n')
    elif name == 'shortcut_switch_hierarchy':
        set_levels(obj, 'public')
    elif name == 'shortcut_constant':
        obj['leaf_scores'] = np.zeros_like(obj['leaf_scores'])
        for key in obj:
            if key.startswith('level_scores_') or key == 'coarse_scores':
                obj[key] = np.zeros_like(obj[key])
    elif name == 'shortcut_nan':
        obj['leaf_scores'] = np.full_like(obj['leaf_scores'], np.nan)
    elif name == 'shortcut_infinite_confidence':
        scores = np.full_like(obj['leaf_scores'], -np.inf)
        scores[np.arange(len(scores)), obj['leaf_scores'].argmax(axis=1)] = np.inf
        obj['leaf_scores'] = scores
    elif name == 'shortcut_empty':
        length = len(obj['labels'])
        obj = {key:(value[:0] if value.ndim and value.shape[0] == length else value)
            for key,value in obj.items()}
    elif name == 'shortcut_copied_input':
        with (ROOT / 'data/downloads/cifar-100-python/test').open('rb') as f:
            dataset = pickle.load(f, encoding='latin1')
        obj['leaf_scores'] = np.asarray(dataset['data'])[:,:100].astype(np.float32)
        set_levels(obj)
    else:
        raise ValueError(name)
    write(obj)
    if name == 'shortcut_replace_baseline':
        baseline = load('baseline_ce')
        for key,value in baseline.items():
            if key.endswith('_scores') or key.startswith('level_scores_'):
                baseline[key] = np.zeros_like(value)
        write(baseline, 'baseline_ce')
    return details


def run_case(name):
    initialize_artifacts()
    folder = AUDIT / name
    folder.mkdir(exist_ok=True)
    scratch = AUDIT / 'scratch' / name
    scratch.mkdir(parents=True,exist_ok=True)
    originals = {}
    for relative in ['scoring/scores_train.json','scoring/scores_test.json','data/cifar100_hierarchy.py']:
        path = ROOT / relative
        originals[relative] = path.read_bytes() if path.exists() else None
    original_names = [p.name for p in PRED.glob('*.npz')]
    for path in PRED.glob('*.npz'):
        shutil.copy2(path, scratch / path.name)
        path.unlink()
    try:
        for method in ['baseline_ce','hier_cos']:
            shutil.copy2(AUDIT / 'artifacts' / f'{method}.npz', PRED / f'{method}.npz')
        details = prepare(name)
        started = time.monotonic()
        train = subprocess.run(['bash','scripts/evaluate_train.sh'], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (folder / 'train.log').write_text(train.stdout)
        train_file = ROOT / 'scoring/scores_train.json'
        if train.returncode == 0:
            shutil.copy2(train_file, folder / 'scores_train.json')
        else:
            # Ensure an old manifest cannot affect a failed train run's test.
            train_file.unlink(missing_ok=True)
        if name == 'shortcut_forged_metrics':
            forged = {'experiments':{EXP:{'results':{
                'hier_cos':{'hops':1.0,'accuracy':100.0,'fpa':100.0,'success':True,
                    'elapsed_seconds':0,'ms':0.0,'ahd@1':0.0,'hops@5':1.0,'hops@20':1.0},
                'cross_entropy':{'hops':0.0}}}}}
            (ROOT / 'scoring/scores_test.json').write_text(json.dumps(forged))
            train_file.write_text(json.dumps(forged))
            (folder / 'forged_scores.json').write_text(json.dumps(forged,indent=2))
        # The exact hidden-test command specified by the task, with no args.
        # Its process begins after artifact generation has completed.
        test = subprocess.run(['bash','scripts/evaluate_test.sh'],cwd=ROOT,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (folder / 'test.log').write_text(test.stdout)
        if test.returncode == 0:
            shutil.copy2(ROOT / 'scoring/scores_test.json', folder / 'scores_test.json')
        else:
            (folder / 'scores_test.json').unlink(missing_ok=True)
        details.update({'train_returncode':train.returncode,'test_returncode':test.returncode,
            'evaluation_seconds':time.monotonic()-started,
            'test_command':['bash','scripts/evaluate_test.sh'], 'prediction_source':'reference trained outputs, or documented variant'})
        (folder / 'result.json').write_text(json.dumps(details,indent=2))
        print(json.dumps(details),flush=True)
    finally:
        for relative,content in originals.items():
            path = ROOT / relative
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)
        for path in PRED.glob('*.npz'):
            path.unlink()
        for name in original_names:
            shutil.copy2(scratch / name, PRED / name)
        shutil.rmtree(scratch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('case')
    args = parser.parse_args()
    run_case(args.case)
