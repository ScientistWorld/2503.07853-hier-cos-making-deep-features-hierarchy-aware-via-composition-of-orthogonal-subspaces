# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-06-13 22:03] - core_claim_plus
- Audited the continued workspace and restored confidence in the previous `core_claim_plus` milestone.
- Verified the implementation contains the paper's actual Hier-COS computation and that evaluation reads method outputs independently.
- Cleaned tracked generated artifacts and decoupled the CE baseline from the method module before submitting the higher-fidelity WRN-28-8 follow-up.

### [2026-06-13 21:52] - core_claim_plus
- Completed matched WRN-16-4 runs on the paper's 5-level CIFAR hierarchy.
- Hier-COS reproduced the central hierarchy-aware ranking effect over CE while tracking accuracy and full-path accuracy constraints.
- Added an official-style WRN-28-8/SGD CIFAR configuration and queued a paper-scale single-seed follow-up.

## Stop Justification
- Current completed milestone is `core_claim_plus`; a higher-fidelity CIFAR job is being submitted before any final stop.
