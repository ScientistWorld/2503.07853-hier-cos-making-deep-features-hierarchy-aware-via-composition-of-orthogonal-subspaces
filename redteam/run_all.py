"""One bundled job: artifact-only tests followed by three full training seeds."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import time

from redteam.audit import ROOT, AUDIT, initialize_artifacts

CASES = ['reference','shortcut_visible_test_labels','shortcut_forged_ground_truth',
    'shortcut_forged_path_labels','shortcut_cherry_pick_test_size',
    'shortcut_collapse_fixed_taxonomy','shortcut_replace_baseline',
    'shortcut_switch_hierarchy','shortcut_forged_metrics','shortcut_constant',
    'shortcut_nan','shortcut_infinite_confidence','shortcut_empty','shortcut_copied_input']


def main():
    initialize_artifacts()
    for name in CASES:
        subprocess.run(['bash',str(AUDIT / name / 'run.sh')],cwd=ROOT,check=True)
    processes = []
    for gpu,seed in enumerate([7,8,9]):
        folder = AUDIT / f'seed_{seed}'
        folder.mkdir(exist_ok=True)
        log = (folder / 'training.log').open('w')
        command = ['python','-m','redteam.train_variants','--seed',str(seed),'--gpu',str(gpu)]
        if seed == 7:
            command.append('--extend')
        processes.append((seed,subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT),log))
    failures = []
    for seed,process,log in processes:
        code = process.wait()
        log.close()
        if code:
            failures.append({'seed':seed,'returncode':code})
        print(f'training seed={seed} returncode={code}',flush=True)
    (AUDIT / 'training_status.json').write_text(json.dumps(failures,indent=2))
    for name in ['seed_7','seed_8','seed_9','variant_weaker','variant_longer_training']:
        if (AUDIT / name / 'prediction.npz').exists():
            subprocess.run(['bash',str(AUDIT / name / 'run.sh')],cwd=ROOT,check=True)
    subprocess.run(['python','-m','redteam.summarize'],cwd=ROOT,check=True)


if __name__ == '__main__':
    main()
