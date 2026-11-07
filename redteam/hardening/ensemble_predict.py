"""Diagnostic only: average Hier-COS projection energies of several trained seeds.

Ensembles are out of bounds for scientists (scoring/CONSTRAINTS.md); this
artifact exists solely to check that genuinely better predictions score higher
on the hardened test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.cifar100_dataset import Cifar100TestImages, Cifar100Train
from data.cifar100_hierarchy import build_hierarchy
from data.prediction_format import save_predictions
from method.hiercos_model import HierCOS
from method.train_hiercos import level_subspace_masks, node_subspace_dims


@torch.no_grad()
def energies(models, loader, device):
    out = []
    for images, *_ in loader:
        images = images.to(device)
        out.append((sum(m(images)["coords"].square() for m in models) / len(models)).cpu())
    return torch.cat(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-root", default=str(ROOT / "data/downloads/cifar-100-python"))
    args = p.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    h = build_hierarchy("paper_l5")
    subspaces = node_subspace_dims(h, "composed")
    models = []
    for path in args.checkpoints:
        model = HierCOS(h.num_nodes, [subspaces[i] for i in h.leaf_node_indices],
                        [subspaces[i] for i in h.level_node_indices[0]], "official_wrn28_8")
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
        models.append(model.to(device).eval())
    leaf_mask = models[0].leaf_mask.cpu()
    masks = level_subspace_masks(h, subspaces)
    outputs = []
    for dataset in (Cifar100Train(args.data_root, "val"), Cifar100TestImages(args.data_root)):
        e = energies(models, DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2), device)
        leaf = torch.sqrt((e @ leaf_mask.t()).clamp_min(1e-12)).numpy()
        levels = [torch.sqrt((e @ m.t()).clamp_min(1e-12)).numpy() for m in masks]
        outputs.append((leaf, levels))
    save_predictions(args.output, outputs[0][0], outputs[0][1], outputs[1][0], outputs[1][1])
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
