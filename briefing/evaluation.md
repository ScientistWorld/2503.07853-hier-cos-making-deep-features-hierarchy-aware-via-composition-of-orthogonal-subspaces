# Evaluation

## Metrics

The evaluator reads saved leaf-class score arrays and labels from `outputs/predictions/*.npz`.

Accuracy and FPA are percentages where higher is better. FPA requires both the fine prediction and its associated coarse prediction to be correct. Mistake Severity is the average LCA distance over misclassified samples, where lower is better. AHD@k averages the LCA distance of the top-k predicted leaves over all samples, where lower is better. HOPS, HOPS@5, and HOPS@20 compare the predicted ranking against the taxonomy-induced preference order; higher is better.

## Baselines and Targets

The reference target is the paper's CIFAR-100 result for the flat cross-entropy baseline and the proposed hierarchy-aware classifier in Table 2, plus the CIFAR-100 FPA value from Table 4. The primary runnable benchmark uses the released 5-level CIFAR hierarchy table associated with the paper.

## Evaluation Protocol

Run training scripts to write one `.npz` prediction artifact per method, then run `bash scripts/evaluate.sh outputs/predictions`. The evaluator is method-agnostic and does not import from `method/`. For fair comparison, evaluate on the same CIFAR-100 test split and use the `paper_l5` hierarchy unless explicitly running an ablation.
