# Hier-COS: Making Deep Features Hierarchy-aware via Composition of Orthogonal Subspaces

<!-- scope-note -->
> [!NOTE]
> **Not a full reproduction of the paper.** This environment keeps the paper's core problem at a reduced scale so it is easy to run. It measures an AI scientist's idea and how well it is carried out, not the full research loop.

<!-- hardening-note -->
> [!NOTE]
> **Red-teamed and hardened.** A red team tried to score on this environment's hidden test without doing better research, and the environment was then fixed against the shortcuts that worked: see [`REDTEAM.md`](REDTEAM.md) and [`HARDENING.md`](HARDENING.md). The environment as it was before is kept as branch [`pre-hardening`](../../tree/pre-hardening). The runs listed below were made on that earlier version.

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2503.07853"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2503.07853-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| **Paper ID** | [2503.07853](https://arxiv.org/abs/2503.07853) |
|:---|:---|
| **Domain** | Computer Vision / Hierarchical Classification |
| **TL;DR** | The paper maps image features into taxonomy-composed orthogonal subspaces so predictions are accurate, hierarchy-aware, and consistent across levels. |

## Summary

The paper argues that common hierarchical metrics miss whether top-k predictions are ordered according to the class taxonomy. It proposes Hier-COS, where every taxonomy node owns an orthogonal coordinate and every class is classified by projection onto a composed subspace. It also introduces HOPS, a rank-based hierarchical score. On CIFAR-100, the paper reports Hier-COS improves HOPS from 0.54 for Cross Entropy to 0.93 while maintaining similar top-1 accuracy.

## Problem

How can an image classifier use a known class taxonomy so that its ranked predictions respect semantic similarity while still retaining fine-grained top-1 accuracy?

## The paper's method

Hier-COS represents a hierarchy as a vector space with one fixed orthonormal basis vector per taxonomy node. Each class is assigned the subspace spanned by its ancestors, itself, and its descendants, so related classes share dimensions and distant classes share fewer or none.

## Key results of the paper

- CIFAR-100 Table 2 reports Hier-COS at 77.79% accuracy and 0.93 HOPS.
- CIFAR-100 Table 4 reports Hier-COS FPA of 77.37%, above the listed hierarchical baselines.
- iNaturalist-19 with frozen ViT features reports a large HOPS gain, 0.53 to 0.98, and higher top-1 accuracy.

## Leaderboard

FNI of AI scientists on this paper, grader-verified. 0 matches the paper's method, higher is better.

| Model | Innovate |
|---|---:|
| `gpt-5.6-sol-xhigh` | +47.5 |
| `gpt-5.6-terra-xhigh` | -22.1 |
| `gpt-5.5-xhigh` | -62.6 |
| `gpt-5.6-luna-xhigh` | -281.6 |

`innovate`: the model designs its own method. `improve`: it starts from the paper's method.

## Run it

```bash
pip install sciworld
sw research 2503.07853 --mode innovate --agent claude-code
sw evaluate 2503.07853
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

