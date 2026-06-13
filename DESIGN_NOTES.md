# Train/Test Evaluation Design Notes

## Checked

- Confirmed `scripts/evaluate_train.sh` invokes `eval.train.evaluate` and writes `scoring/scores_train.json`.
- Confirmed `scripts/evaluate_test.sh` invokes `eval.test.evaluate` and writes `scoring/scores_test.json`.
- Confirmed test scoring reads method prediction artifacts from disk rather than regenerating method outputs.
- Confirmed train and test evaluators do not cross-import from each other.
- Confirmed no `scoring/scores_train.json` or `scoring/scores_test.json` stubs are present.
- Confirmed `bash -n` passes for both train/test evaluation scripts.
- Checked `scoring/reference.json` for paper-agreement metric names and wall-clock metrics; none are present.

## Changed

- Added metric descriptions to all active entries in `scoring/reference.json`; the previous entries had coefficients and directions but no descriptions.
- Added train/test scoring for `macro_accuracy`, `top5_accuracy`, `coarse_accuracy`, and `fine_coarse_consistency`.
- Reweighted accuracy and hierarchy-distance side metrics slightly so HOPS remains primary while additional constraints make metric gaming harder.
- Made `fine_coarse_consistency` require an explicit submitted `coarse_scores` artifact; missing explicit coarse predictions score zero for that metric.

## Rationale

The original split already used a complementary fine-label-stratified 50/50 slice of the CIFAR-100 test predictions, which is a good lightweight anti-overfit design for this paper. The added metrics sharpen the distinction between genuine hierarchy-aware improvement and shortcut behavior:

- `macro_accuracy` catches improvements concentrated in a small subset of classes.
- `top5_accuracy` checks that the true fine label remains near the top of the ranked list.
- `coarse_accuracy` checks utility at the superclass level.
- `fine_coarse_consistency` catches artifacts whose explicit coarse prediction contradicts the parent of the submitted fine prediction.

## Runner Concerns

- The train/test split is still drawn from the same CIFAR-100 test distribution, so it detects visible-slice overfitting but does not add a new OOD domain.
- The new metrics are computed from existing `.npz` prediction artifacts and should be cheap, but the next runner should confirm generated score files include all metric keys listed in `reference.json`.
- `fine_coarse_consistency` will penalize future methods that submit only `leaf_scores`; this is intentional for the paper's multi-level consistency claim, but from-scratch agents need the artifact contract documented clearly.
