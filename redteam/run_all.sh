#!/bin/bash
set -euo pipefail
cd /home/user
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Keep the two GPUs busy with independent training; predictions and scoring
# are serialized by run_case.py and never share a mutable candidate directory.
worker_a() {
  python redteam/train_variant.py checkpoint --gpu 0 > redteam/reference_checkpoint/train.log 2>&1
  bash redteam/reference_checkpoint/run.sh
  for seed in 7 9; do
    python redteam/train_variant.py seed --seed "$seed" --gpu 0 > "redteam/reference_seed_$seed/train.log" 2>&1
    bash "redteam/reference_seed_$seed/run.sh"
    if [ "$seed" = 7 ]; then bash redteam/variant_less_training/run.sh; fi
  done
}
worker_b() {
  python redteam/train_variant.py bn --gpu 1 > redteam/variant_bn_recalibration/train.log 2>&1
  bash redteam/variant_bn_recalibration/run.sh
  python redteam/train_variant.py extended --gpu 1 > redteam/variant_more_training/train.log 2>&1
  bash redteam/variant_more_training/run.sh
  python redteam/train_variant.py seed --seed 8 --gpu 1 > redteam/reference_seed_8/train.log 2>&1
  bash redteam/reference_seed_8/run.sh
}

for case in reference_artifacts variant_flat_baseline variant_taxonomy_rerank \
    shortcut_test_labels shortcut_constant_self_labels shortcut_select_correct \
    shortcut_path_labels shortcut_omit_leaf_path_level shortcut_baseline_replacement \
    shortcut_taxonomy_artifact shortcut_write_scores shortcut_constant shortcut_nan \
    shortcut_infinity shortcut_empty shortcut_method_alias; do
  bash "redteam/$case/run.sh"
done
worker_a &
pid_a=$!
worker_b &
pid_b=$!
status=0
wait "$pid_a" || status=1
wait "$pid_b" || status=1
printf '%s\n' "$status" > redteam/job_exit_status.txt
exit "$status"
