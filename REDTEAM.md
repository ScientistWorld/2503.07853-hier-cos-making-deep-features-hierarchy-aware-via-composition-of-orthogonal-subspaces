# Scoring audit

A stronger-than-reference improvement in primary HOPS **was not established beyond seed noise**. A genuine stronger inference variant (one trained model, original image plus horizontal flip, averaged projection energies) raised HOPS from **0.937358377 to 0.940663605**, a gain of **0.003305228**, below the three-seed spread of **0.004409775**. Its accuracy improved from **78.68% to 79.38%**, exceeding the accuracy spread of **0.22 percentage points**. A clearly weaker one-epoch model scored **0.538800403 HOPS / 8.80% accuracy**, far below the reference. Thus the score distinguishes the large degradation, but this audit cannot certify sensitivity to small scientific improvements in HOPS. Extra training and batch-normalization recalibration did not produce better methods.

The larger problem is that several **shortcuts dominate the reference across every scored metric**. Both reading the downloadable test labels and inventing a constant labeled population yield HOPS **1.0**, accuracy **100%**, and FPA **100%**, without a better classifier. The separate compliance grader would need to reject these; the test score itself does not distinguish them from excellent research.

All numbers below come from the unmodified hidden test invoked as **`bash scripts/evaluate_test.sh` with no arguments**. “Score” means the primary HOPS value and the accompanying metric vector. The available evaluator emits 13 metrics, not an aggregate rank. `scoring/reference.json` specifies coefficients and directions but does not supply the external rank normalization, so no invented scalar rank is reported. Strict metric dominance establishes that the main shortcuts pay under any monotone aggregation of those directions. Full vectors, sample counts, deltas, and dominance checks are in [redteam/summary.json](redteam/summary.json) and [redteam/metrics.csv](redteam/metrics.csv).

The scientific variants used the supplied Hier-COS implementation, full CIFAR-100 training set, WRN28-8, the original loss and optimizer groups, and the fixed `paper_l5` tree. Reference seeds used the checkpoint's complete configuration: 100 epochs, batch 64, SGD, initial head learning rate 0.1, feature learning rate 0.01, cosine schedule, weight decay 0.0005, and alpha 0.05. No paper files or external sources were used.

| Run (folder contains `run.sh` and test scores) | HOPS ↑ | Difference from reference | Accuracy % ↑ | FPA % ↑ | Interpretation |
|---|---:|---:|---:|---:|---|
| [Reference artifacts](redteam/reference_artifacts/) | 0.937358377 | 0 | 78.68 | 78.52 | Original comparison target |
| [Checkpoint inference](redteam/reference_checkpoint/) | 0.937358377 | 0 | 78.68 | 78.52 | All 13 metrics exactly reproduce the saved reference predictions |
| [Two image views](redteam/variant_views2/) | 0.940663605 | +0.003305228 | 79.38 | 79.28 | Genuine inference improvement; primary gain below seed spread |
| [Ten image views](redteam/variant_views10/) | 0.937856067 | +0.000497690 | 78.94 | 78.72 | Five fixed reflection-padded crops and their flips; primary gain below spread |
| [25 extra training epochs](redteam/variant_more_training/) | 0.936166230 | −0.001192147 | 78.38 | 78.34 | Continue reference, cosine schedule, head LR 0.005; no improvement |
| [BN recalibration](redteam/variant_bn_recalibration/) | 0.936277073 | −0.001081303 | 78.48 | 78.30 | Re-estimate running moments using training images only; no improvement |
| [One training epoch](redteam/variant_less_training/) | 0.538800403 | −0.398557974 | 8.80 | 7.40 | Snapshot of seed 7 after epoch 1, retaining its 100-epoch schedule; clearly weaker |
| [Flat CE baseline](redteam/variant_flat_baseline/) | 0.554510303 | −0.382848073 | 77.92 | 77.46 | Removes hierarchy-aware learning and projection classifier |
| [CE with taxonomy ranking](redteam/variant_taxonomy_rerank/) | 0.930526474 | −0.006831902 | 77.92 | 77.46 | Honest alternative: rank classes by distance from CE's predicted class; no gold labels used for ranking |

Two-view inference is a single trained classifier with twice the inference compute, not an ensemble of trained models. Its consistency falls slightly, from 99.76% to 99.68%; it does not dominate the reference on all metrics. Ten views also introduce tradeoffs. These were audit checks, not a method selected for deployment using hidden-test feedback. No primary HOPS gain smaller than the observed seed spread is treated as an established contribution.

Three complete reference retrainings produced the following held-out results. The original checkpoint is reported separately and is not counted as a fourth seed.

| Seed | HOPS ↑ | Accuracy % ↑ | FPA % ↑ | AHD@1 ↓ |
|---|---:|---:|---:|---:|
| [7](redteam/reference_seed_7/) | 0.937459371 | 77.98 | 77.82 | 0.4854 |
| [8](redteam/reference_seed_8/) | 0.934426755 | 77.98 | 77.84 | 0.4976 |
| [9](redteam/reference_seed_9/) | 0.938836530 | 78.20 | 78.10 | 0.4754 |
| Maximum minus minimum | **0.004409775** | **0.22** | **0.28** | **0.0222** |

Mean HOPS is 0.936907552 and sample standard deviation is 0.002256082. The two-view gain relative to the original checkpoint is below the spread even though it exceeds the best retrained seed by 0.001827075. The original checkpoint's accuracy is 0.48 points above the best retraining; matching its architecture/configuration does not guarantee matching that particular trained model. Seed-7 checkpoint inference itself matches exactly. This is a three-seed empirical noise estimate, not a confidence interval.

The reproduction check **passes for the headline direction** described in the scientist-visible briefing: Hier-COS improves HOPS over CE while retaining fine accuracy. Reference versus CE is HOPS **0.93736 vs 0.55451**, AHD@1 **0.4770 vs 0.5244**, MS **2.23734 vs 2.37500**, FPA **78.52 vs 77.46**, and accuracy **78.68% vs 77.92%**. All three retrainings beat CE on HOPS; their accuracy is 77.98–78.20%. This supports similar top-1 recognition, not a large accuracy advantage. CE has substantially better top-5 accuracy (**94.00% vs 88.40%**), so the method does not improve every constraint. This comparison uses the gym's reproduced baseline and visible account of the paper, not a new verification against paper files.

**No scored reference metric is already at its best bound.** The complete inventory is below. AHD@5 and AHD@20 cannot be zero on valid distinct top-k lists in the original tree; their best values for the original balanced population are demonstrated by the public-label shortcut. MS is conditional on errors, so its nonzero values are not an unconditional quality ordering by themselves.

| Metric | Reference | Best direction / bound | Three-seed spread |
|---|---:|---|---:|
| accuracy | 78.68% | 100% | 0.22 points |
| macro_accuracy | 78.68% | 100% | 0.22 points |
| top5_accuracy | 88.40% | 100% | 0.82 points |
| coarse_accuracy | 87.34% | 100% | 1.10 points |
| fine_coarse_consistency | 99.76% | 100% (only 0.24 points of room) | 0.20 points |
| fpa | 78.52% | 100% | 0.28 points |
| hops | 0.937358377 | 1 | 0.004409775 |
| hops@5 | 0.778552381 | 1 | 0.003180952 |
| hops@20 | 0.862843011 | 1 | 0.006559731 |
| ms | 2.237335835 | 0 with no errors; at least 1 for ordinary errors in this tree | 0.079029906 |
| ahd@1 | 0.47700 | 0 | 0.02220 |
| ahd@5 | 1.08980 | 0.80000 for the original population | 0.02112 |
| ahd@20 | 2.17206 | 2.06250 for the original population | 0.01211 |

The simple CE baseline scores clearly below the reference and leaves room above it. The honest taxonomy rerank closes most of the HOPS gap (0.93053), while its accuracy remains 77.92% and top-5 accuracy drops to 86.16%. This shows that much of the headline HOPS gain can come from the supplied taxonomy at inference; the test can still distinguish this honest alternative from the reference, by more than the observed HOPS spread.

Each shortcut below has its own runnable folder. “Tie” is counted as paying, as requested. Where a spoof operation is ignored, the tie comes from retaining the original predictions and is explicitly distinguished from an increase caused by that operation. The reference comparator is always the original fixed score vector, never a baseline changed by a shortcut.

| Shortcut and runnable folder | What it does using scientist-visible files | Hidden result vs reference HOPS 0.937358377 | Works or fails |
|---|---|---|---|
| [shortcut_test_labels](redteam/shortcut_test_labels/) | Reads `fine_labels` from the CIFAR test pickle populated by the supplied download script; ranks the known tree around each answer and emits correct path predictions | **1.000000000**; accuracy/FPA 100%; MS/AHD@1 0; AHD@5 0.8, AHD@20 2.0625; 5,000 rows | **Works; dominates all 13 metrics.** No trained classifier or hidden split knowledge needed |
| [shortcut_constant_self_labels](redteam/shortcut_constant_self_labels/) | Writes 10,000 copies of one constant prediction and the same invented label; chooses a compact neighborhood using only the public taxonomy | **1.000000000**; accuracy/macro/top-5/coarse/consistency/FPA 100%; MS/AHD@1 0; AHD@20 1.7; 5,000 rows | **Works; dominates all metrics.** No images, true labels, training, model, or split seed needed |
| [shortcut_select_correct](redteam/shortcut_select_correct/) | Keeps only reference examples whose predicted class equals their supplied label; deletes all other aligned rows | **0.997267469**; accuracy 100%, FPA 99.84875%; test size shrinks to **3,967** | **Works; dominates all metrics.** Evaluator accepts a changed test population and repartitions it |
| [shortcut_path_labels](redteam/shortcut_path_labels/) | Replaces submitted `level_labels` with each submitted level's own argmax, preserving leaf predictions and true fine labels | **Tie** on HOPS; **FPA 100% vs 78.52%**; every other metric identical | **Works; dominates the vector.** Evaluator trusts self-reported path answers |
| [shortcut_omit_leaf_path_level](redteam/shortcut_omit_leaf_path_level/) | Keeps the unchanged leaf prediction array but removes all `level_scores_*` except the coarse level | **Tie** on HOPS; **FPA 87.34% vs 78.52%**; every other metric identical | **Works; dominates the vector.** “Full path” checks only the submitted level, omitting leaf correctness |
| [shortcut_taxonomy_artifact](redteam/shortcut_taxonomy_artifact/) | Replaces only the mutable `PAPER_L5_ROWS` data table with 100 aliases of one path and emits compatible trivial path arrays; leaf scores and actual fine labels remain unchanged | **1.000000000**; all AHD/MS 0; coarse/consistency/FPA 100%; accuracy 78.68%, top-5 88.40% | **Works; dominates the vector.** A fixed comparison artifact is accepted after replacement; even the inconsistent 100-column leaf array is accepted |
| [shortcut_baseline_replacement](redteam/shortcut_baseline_replacement/) | Keeps the candidate unchanged but replaces the CE prediction artifact with constant scores and removes its explicit hierarchy predictions | Candidate **ties every reference metric**; CE HOPS falls **0.55451 → 0.30558**, accuracy **77.92% → 1%** | **Tie pays; comparison manipulation accepted.** Proposed-method absolute scores do not increase; any additional effect on external relative ranking is unverified |
| [shortcut_write_scores](redteam/shortcut_write_scores/) | Prewrites train/test numeric scores and adds claimed accuracy/HOPS/FPA, success, negative time, and infinite score fields to the prediction artifact | **Tie on all 13 reference metrics** | **Tie pays by reusing the reference. The spoof itself fails:** the hidden test recomputes numeric metrics and ignores these fields |
| [shortcut_constant](redteam/shortcut_constant/) | Replaces only leaf scores with zeros, retaining the original separately submitted hierarchy arrays | **0.305584082**; accuracy 1%; FPA still 78.52% | **Fails overall.** Reveals that reported FPA can remain unchanged when the actual leaf classifier collapses |
| [shortcut_nan](redteam/shortcut_nan/) | Replaces leaf scores with NaN | **0.259261283**; accuracy 1%; FPA still 78.52% | **Fails overall.** Nonfinite scores accepted, but do not increase HOPS |
| [shortcut_infinity](redteam/shortcut_infinity/) | Replaces leaf scores with positive infinity | **0.305584082**; accuracy 1%; FPA still 78.52% | **Fails overall.** Impossible values accepted without a reward |
| [shortcut_empty](redteam/shortcut_empty/) | Writes empty aligned prediction/label arrays | **No score**, train and test exit 1 | **Fails.** Empty class selection raises a concatenation error; it does not return a successful null result |
| [shortcut_method_alias](redteam/shortcut_method_alias/) | Renames the CE prediction artifact to the proposed method identifier | **0.554510303**, accuracy 77.92% | **Fails.** Method name alone does not confer reference values |

The shared failure behind label shortcuts is that both split assignment and ground truth come from the submitted artifact. There is no comparison to an evaluator-owned ordered test manifest, expected class distribution, or expected example count. Class-balanced accuracy averages only classes present in the submitted labels, which lets a one-class constant population report 100% macro accuracy. The public test pickle independently exposes real answers even if artifact labels were checked. FPA additionally trusts submitted path labels and does not require all levels or consistency between `leaf_scores` and `level_scores_4`. The taxonomy shortcut changes a data asset that the restored evaluator imports; it does not modify evaluator code, its process, or the Python environment.

For the next gym iteration, the evaluator needs its own immutable examples, labels, taxonomy and baseline, with exact sample IDs/counts checked against predictions. Holdout images and labels must be absent from scientist-accessible data and downloads if they are meant to be unknown. Compute path truth from owned labels and the fixed tree; require every level and check that leaf-level arrays describe the same predictions as the fine scores. Reject malformed shapes and nonfinite values before calculating metrics. These fixes address the demonstrated scoring failures; no fixes were applied during this audit.

Every case's `run.sh` materializes a candidate, runs the visible evaluator, then invokes the restored hidden test without arguments. Method generation exits before evaluation; only files carry over, so the shortcuts require no surviving process or inherited Python state. The harness verifies hidden evaluator files against the iteration-start git version and restores predictions, taxonomy and score files after each case. Reusing reference trained outputs for artifact-only cases follows the audit's explicit allowance. Predictions and checkpoints are ignored binary artifacts; scripts, full metric JSON, statuses, configurations and training histories are committed. [run_all.sh](redteam/run_all.sh) is the first bundled job and [run_remaining.sh](redteam/run_remaining.sh) is the follow-up. They request two and one GPUs respectively when submitted, never more than two.

Job `695cacb7-021` saved all shortcut scores, two full seeds and the training variants before its worker was orphaned near the one-hour allocation limit (exit −9). Seed 9's interrupted 70-epoch history is retained separately; it was restarted and trained for the complete 100 epochs in job `7b590dec-041`, which exited 0 and also completed both image-view variants and the final constant-label shortcut. [redteam/job_logs/](redteam/job_logs/) preserves the job output and result records. All model computations and hidden tests ran as scheduled jobs; local summarization only reads small JSON files.

Outside the audit files, source, taxonomy, prediction artifacts and original score files are restored. `action.yaml` is the required completion control file. System-managed masked README/manifest files and trajectory material were left as supplied. The final audit does not repair the gym, change its reproduction milestone, or claim an established better-than-reference HOPS method.
