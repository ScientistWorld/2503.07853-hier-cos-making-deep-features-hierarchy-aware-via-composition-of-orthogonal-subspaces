#!/bin/bash
# Hardening job: smoke-test the new pipeline, retrain the reference and the CE
# baseline on the hardened protocol (45K train / 5K val / 10K hidden test),
# train better/weaker variants, freeze the baseline into the evaluators, score
# the reference, then run every variant and shortcut through the real
# evaluators. Every step is skipped when its artifact already exists, so a
# killed job resumes. Uses cuda:0 and cuda:1 of the job's own allocation.
set -uo pipefail
cd /home/user
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1

H=redteam/hardening
LOG=$H/train_logs
RUNS=$H/runs
mkdir -p "$LOG" "$RUNS/predictions" "$RUNS/ckpt" outputs/predictions outputs/baseline checkpoints

# ── 0. smoke: tiny end-to-end run of training, prediction, freezing, both evaluators
if [ ! -f "$H/smoke/ok" ]; then
  rm -rf "$H/smoke"; mkdir -p "$H/smoke"
  python -m baseline.train_ce --epochs 1 --train-limit 256 --run-name cross_entropy --output-dir "$H/smoke/baseline" --device cuda:0 > "$LOG/smoke_ce.log" 2>&1 \
  && python -m method.train_hiercos --epochs 1 --train-limit 256 --run-name smoke_hier_cos --output-dir "$H/smoke/predictions" --checkpoint-dir "$H/smoke/ckpt" --device cuda:0 > "$LOG/smoke_hiercos.log" 2>&1 \
  && python -m baseline.freeze_fixed_baseline --source "$H/smoke/baseline/cross_entropy.npz" > "$LOG/smoke_freeze.log" 2>&1 \
  && python -m eval.train.evaluate --prediction-dir "$H/smoke/predictions" --output "$H/smoke/scores_train.json" > "$LOG/smoke_eval_train.log" 2>&1 \
  && python -m eval.test.evaluate --prediction-dir "$H/smoke/predictions" --output "$H/smoke/scores_test.json" > "$LOG/smoke_eval_test.log" 2>&1 \
  && touch "$H/smoke/ok"
  if [ ! -f "$H/smoke/ok" ]; then
    echo "SMOKE FAILED"; tail -n 40 "$LOG"/smoke_*.log; exit 1
  fi
  echo "smoke ok"
fi

hiercos() {  # name device extra-args...  -> $RUNS/predictions/name.npz
  local name=$1 dev=$2; shift 2
  [ -f "$RUNS/predictions/$name.npz" ] && return 0
  python -m method.train_hiercos --run-name "$name" --output-dir "$RUNS/predictions" --checkpoint-dir "$RUNS/ckpt" --device "$dev" "$@" > "$LOG/$name.log" 2>&1 \
    || echo "FAILED $name"
}

lane_a() {
  if [ ! -f outputs/baseline/cross_entropy.npz ]; then
    python -m baseline.train_ce --seed 7 --run-name cross_entropy --device cuda:0 > "$LOG/cross_entropy_seed7.log" 2>&1 || echo "FAILED cross_entropy"
  fi
  hiercos reference_seed8 cuda:0 --seed 8
  hiercos weaker_decreasing_weights cuda:0 --seed 7 --level-weights decreasing
  hiercos weaker_uniform_weights cuda:0 --seed 7 --level-weights uniform
  hiercos weaker_epochs5 cuda:0 --seed 7 --epochs 5
}

lane_b() {
  if [ ! -f outputs/predictions/hier_cos.npz ]; then
    if [ -f checkpoints/hier_cos.pt ]; then
      python -m method.train_hiercos --seed 7 --run-name hier_cos --load-checkpoint checkpoints/hier_cos.pt --epochs 0 --device cuda:1 > "$LOG/hier_cos_seed7_predict.log" 2>&1 || echo "FAILED hier_cos predict"
    else
      python -m method.train_hiercos --seed 7 --run-name hier_cos --device cuda:1 > "$LOG/hier_cos_seed7.log" 2>&1 || echo "FAILED hier_cos"
    fi
  fi
  hiercos better_tta_flip cuda:1 --seed 7 --load-checkpoint checkpoints/hier_cos.pt --epochs 0 --tta flip
  hiercos reference_seed9 cuda:1 --seed 9
  hiercos weaker_single_subspace cuda:1 --seed 7 --subspace single
}

lane_a &
pid_a=$!
lane_b &
pid_b=$!
wait "$pid_a"; wait "$pid_b"

if [ ! -f "$RUNS/predictions/better_ensemble_diag.npz" ] && [ -f "$RUNS/ckpt/reference_seed8.pt" ]; then
  python $H/ensemble_predict.py checkpoints/hier_cos.pt "$RUNS/ckpt/reference_seed8.pt" \
    --output "$RUNS/predictions/better_ensemble_diag.npz" --device cuda:0 > "$LOG/better_ensemble_diag.log" 2>&1 || echo "FAILED ensemble"
fi

# ── reference scores (reference run alone) and the frozen baseline
python -m baseline.freeze_fixed_baseline --source outputs/baseline/cross_entropy.npz > "$LOG/freeze.log" 2>&1 || { echo "FAILED freeze"; exit 1; }
rm -f scoring/scores_train.json scoring/scores_test.json
bash scripts/evaluate_train.sh > "$LOG/reference_eval_train.log" 2>&1 || echo "FAILED evaluate_train"
bash scripts/evaluate_test.sh > "$LOG/reference_eval_test.log" 2>&1 || echo "FAILED evaluate_test"
bash scripts/evaluate.sh > "$LOG/reference_eval.log" 2>&1 || echo "FAILED evaluate"

# ── every variant and shortcut through the real evaluators
python $H/build_cases.py > "$LOG/build_cases.log" 2>&1 || { echo "FAILED build_cases"; tail -n 30 "$LOG/build_cases.log"; exit 1; }
python $H/score_cases.py > "$LOG/score_cases.log" 2>&1 || { echo "FAILED score_cases"; tail -n 30 "$LOG/score_cases.log"; exit 1; }
grep -h "FAILED" "$LOG"/*.log | head -n 20
echo "hardening job done"
