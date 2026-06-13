# Problem Definition

## Research Question

How can an image classifier use a known class taxonomy so that its ranked predictions respect semantic similarity while still retaining fine-grained top-1 accuracy?

## Why It Matters

Flat classifiers treat every wrong label as equally bad. In hierarchical domains such as species, products, or object taxonomies, confusing two nearby classes is preferable to confusing distant classes, and users may also need consistent predictions at coarse and fine levels.

## Success Criteria

A successful system should classify fine-grained labels accurately, make less severe mistakes when it is wrong, rank semantically closer alternatives above distant ones, and produce coarse-level predictions that agree with the fine-level prediction path.
