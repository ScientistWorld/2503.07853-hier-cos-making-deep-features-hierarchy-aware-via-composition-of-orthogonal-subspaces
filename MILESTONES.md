# Reproduction Milestones

**Current: secondary_claims**

## Progress Log

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
- Current completed milestone is `secondary_claims`. The central CIFAR-100 Table 2 result and the Table 4 FPA constraint are reproduced with the paper's architecture family; higher milestones would require multi-seed reporting and the additional iNaturalist-19/tieredImageNet-H experiment families.
