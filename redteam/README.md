# Scoring audit runs

Run an individual case with `bash redteam/<case>/run.sh` inside the gym's
container as a scheduled job. Every case installs its prediction artifacts,
runs the visible evaluator, and invokes exactly `bash scripts/evaluate_test.sh`
with no arguments in a separate process. Each folder keeps `scores_test.json`,
`scores_train.json` when successful, and a `result.json` recording return codes.
Large predictions and checkpoints, and raw logs, are ignored by git.

The first invocation caches the supplied reference and baseline trained outputs
under `redteam/artifacts/`. This reuse is permitted by the audit instructions.
On an installation without trained outputs, produce them first using the
supplied modules, full CIFAR-100 training data, `paper_l5`, WRN-28-8, batch 64,
100 epochs, SGD learning rate 0.1, weight decay 0.0005, seed 7, and alpha 0.05.
Neither hidden evaluation modules nor hidden answers are used to construct any
shortcut. The public CIFAR download contains its own test labels.

`audit.py` snapshots and restores prediction files, scoring files and the
taxonomy source around each run. It does not modify `eval/`, test scripts,
Python paths, installed packages, or evaluator processes. Taxonomy replacement
is an explicit shortcut targeting an editable, nominally fixed problem asset.
All other shortcuts change only prediction or score artifacts.

`run_all.py` bundles artifact-only checks and full-size training seeds 7, 8, 9.
It needs three assigned GPUs; CUDA devices are selected with the Torch API,
without changing `CUDA_VISIBLE_DEVICES`. `train_variants.py` uses the supplied
model, training loss, optimizer and prediction routines. Seed 7 also exports
the 10-epoch weaker checkpoint and continues for 50 additional epochs after
the standard 100, restarting cosine decay at learning rate 0.01. All training
uses the full 50,000 training examples and predictions cover all 10,000 test
examples. The test labels are never used for training these quality variants.

The summary reports the declared primary score, HOPS, and the complete metric
vector. No additional scalar ranking formula is supplied by the evaluation
entry point; no synthetic weighted aggregate is presented as its official rank.
