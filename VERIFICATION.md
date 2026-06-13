# Verification Report

## Initial Assessment

The workspace started at `secondary_claims` with a completed CIFAR-100 reproduction of "Hier-COS: Making Deep Features Hierarchy-aware via Composition of Orthogonal Subspaces." The claimed milestone is justified: the final run uses the paper's 5-level CIFAR hierarchy, an official-style WRN-28-8 setup, the paper's hierarchy-aware training computation, and method-independent evaluation from saved prediction artifacts.

## Issues Found

- Cosmetic: `briefing/evaluation.md` referred to the reference method as "the proposed hierarchy-aware classifier." This was not an algorithm leak, but it was less neutral than ideal for from-scratch agents.
- Cosmetic: ignored local `__pycache__` folders were present after validation commands.

## Fixes Applied

- Reworded the strict method-agnostic evaluation brief to describe the reference result as the paper's best reported hierarchy-aware system, without naming or describing the algorithm.
- Removed local ignored Python bytecode folders. A few read-only `__pycache__` folders under the ignored shared model cache could not be removed, but they are outside committed code and are not referenced by the workspace.
- Verified `scripts/download.sh` is self-contained, writes to `data/downloads/`, contains no `shared/` reference, and its CIFAR-100 URL returns HTTP 200 with the expected archive size.
- Ran `python validate.py --compare`. It passes. The validator emits heuristic warnings for `eval/train` and `eval/test` because they use deterministic random slicing, but both evaluators read saved `.npz` prediction artifacts and do not import or rerun `method/`.

## Final Judgment

The workspace is ready to serve as a research gym. Improve-mode agents have a working method, baseline, reproduction script, and recorded scores. From-scratch agents can use the problem, data, and evaluation surface without method imports or solution-specific data leakage. The scoring rubric emphasizes the primary HOPS metric while preserving constraints for fine-label accuracy, full-path accuracy, hierarchy distance, top-k behavior, and multi-level consistency.

- Milestone: secondary_claims
- Ready for gym use: yes
- Confidence: high
- Key limitation: CIFAR-100 single-seed reproduction only; multi-seed reporting and the paper's other datasets remain future extensions.
