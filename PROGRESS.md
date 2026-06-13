# Progress

## What Works

- Paper read and core algorithm identified: fixed orthonormal node bases, composed ancestor/self/descendant subspaces, weighted path KL loss, sparsity regularizer, and projection-norm inference.
- Implemented CIFAR-100 data loading, a public CIFAR hierarchy, Hier-COS training, a Cross Entropy baseline, and a method-independent evaluator.

## Results

- No compute-job results yet. The first submitted job is a 2-epoch smoke run on 6,000 CIFAR-100 training images and 2,000 test images.

## Remaining

- Run the smoke job and inspect `scoring/scores.json`.
- If Hier-COS improves hierarchical metrics over the baseline, scale training up toward the full CIFAR-100 split.
- If results are weak, tune epoch count, alpha, learning rate, or the transformation/backbone capacity while preserving the Hier-COS procedure.

## Issues

- The official repository exists at `https://github.com/Depanshu-Sani/Hier-COS`, but it has no license file. I did not vendor it into this redistributable benchmark.

## Deviations from Paper

- Paper setup: CIFAR-100 uses a 5-level hierarchy from prior work, WideResNet, 45K train / 5K val / 10K test, and 5-seed reporting. Current compact setup: public CIFAR-100 20-coarse/100-fine hierarchy, a small CNN trained from scratch, and a first smoke run on a subset. This scales resources and uses a public hierarchy asset; it does not replace the Hier-COS algorithm.
