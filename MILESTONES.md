# Reproduction Milestones

**Current: secondary_claims**

## Progress Log

### [2026-11-07] - secondary_claims (hardening)
- Hardened the hidden test. Test labels are removed from scientist data; labels, taxonomy and the CE baseline are evaluator-owned; the evaluators check artifact shapes and finiteness and fail implausible runs; the full 10K test set is scored. The paper's 45K/5K split is used with best-validation checkpoint selection.
- Retrained the reference (Hier-COS seed 7) and the CE baseline from the final code. On the hidden test: HOPS 0.935 vs 0.549, accuracy 77.23 vs 77.33, FPA 77.12 vs 76.87.
- All major red-team shortcuts now fail or tie. Genuine improvements score above the seed spread and the paper's ablations score below. See HARDENING.md.

### [2026-06-13 23:21] - secondary_claims
- Completed final verification for gym readiness.
- Fixed method/problem separation leakage in the CIFAR hierarchy data utilities and method-agnostic documentation.
- Aligned top-level `scores.json` and `eval/metrics.py` with every nonzero metric coefficient in `reference.json`.

### [2026-06-13 22:54] - secondary_claims
- Completed the official-style single-seed CIFAR-100 WRN-28-8 run on the paper's 5-level hierarchy.
- Reproduced the paper's primary hierarchy-aware ranking effect: HOPS 0.940 for Hier-COS versus 0.556 for CE, close to the reported 0.930 versus 0.540.
- Reproduced the key constraint side: Hier-COS kept comparable or better accuracy/FPA, reaching 78.85% accuracy and 78.73 FPA versus CE at 78.50% and 78.04.

### [2026-06-13 22:03] - core_claim_plus
- Audited the continued workspace and restored confidence in the previous `core_claim_plus` milestone.
- Verified the implementation contains the paper's actual Hier-COS computation and that evaluation reads method outputs independently.
- Cleaned tracked generated artifacts and decoupled the CE baseline from the method module before submitting the higher-fidelity WRN-28-8 follow-up.

### [2026-06-13 21:52] - core_claim_plus
- Completed matched WRN-16-4 runs on the paper's 5-level CIFAR hierarchy.
- Hier-COS reproduced the central hierarchy-aware ranking effect over CE while tracking accuracy and full-path accuracy constraints.
- Added an official-style WRN-28-8/SGD CIFAR configuration and queued a paper-scale single-seed follow-up.

## Stop Justification
- Completed at milestone `secondary_claims`.
