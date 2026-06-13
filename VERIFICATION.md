# Verification Report

## Initial Assessment

The workspace already had a completed `secondary_claims` reproduction. The claimed milestone is justified: the final CIFAR-100 run uses the paper's 5-level hierarchy and an official-style WRN-28-8 setup, with reproduced HOPS, accuracy, AHD, MS, and FPA close to the paper's reported CIFAR-100 values.

## Issues Found

- Minor: `data/cifar100_hierarchy.py` exposed method-specific subspace convenience fields. This leaked solution structure into a problem/data file that from-scratch agents may see.
- Minor: method-agnostic documentation named the proposed method/repository in `briefing/evaluation.md` and referenced a reproduction-specific backbone in `scoring/CONSTRAINTS.md`.
- Critical for scoring hygiene: `reference.json` had nonzero coefficients for `macro_accuracy`, `top5_accuracy`, `coarse_accuracy`, and `fine_coarse_consistency`, but top-level `scores.json` did not include those metrics.
- Cosmetic: generated `__pycache__` folders were present in the workspace.

## Fixes Applied

- Moved hierarchy-derived subspace composition out of `data/` and into `method/train_hiercos.py`.
- Reworded `briefing/evaluation.md` and `scoring/CONSTRAINTS.md` to keep from-scratch-facing documentation method-safe.
- Updated `eval/metrics.py` to compute all nonzero-scored metrics used by `reference.json`.
- Backfilled `scoring/scores.json` with the missing metrics from the already recorded complementary train/test slices.
- Removed generated `__pycache__` directories.
- Verified the CIFAR-100 download URL with `curl -I -L --max-time 15`; it returns HTTP 200 with the expected archive size.
- Ran `python validate.py --compare` and Python syntax checks. `validate.py` passes; it reports heuristic warnings for train/test split files because they use deterministic random slicing, but those files read saved prediction artifacts and do not import or rerun `method/`.

## Final Judgment

The workspace is ready to serve as a research gym. Improve-mode agents can run the provided method and compare against the reproduced scores. From-scratch agents can use the problem/data/eval surface without seeing method-specific representation details in the strict method-agnostic files. Scoring now includes every metric with a nonzero coefficient for both reproduced methods.

- Milestone: secondary_claims
- Ready for gym use: yes
- Confidence: high
- Key limitation: single-seed CIFAR-100 reproduction only; additional datasets and multi-seed reporting remain future extensions.
