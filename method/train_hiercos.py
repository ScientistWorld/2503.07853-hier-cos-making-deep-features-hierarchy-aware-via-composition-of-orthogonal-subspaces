from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.cifar100_dataset import Cifar100TestImages, Cifar100Train, maybe_subset
from data.cifar100_hierarchy import Cifar100Hierarchy, build_hierarchy
from data.prediction_format import save_predictions
from method.hiercos_model import HierCOS, hiercos_loss

# Taxonomy levels scored by the benchmark besides the leaves (levels 0..3).
NON_LEAF_LEVELS = 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/downloads/cifar-100-python")
    # Defaults are the paper's CIFAR-100 configuration (HAFrame/Hier-COS WRN-28-8 recipe).
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--alpha", type=float, default=5e-2)
    parser.add_argument("--backbone", choices=["small", "wrn16_4", "official_wrn28_8"], default="official_wrn28_8")
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="sgd")
    parser.add_argument("--hierarchy", choices=["public", "paper_l5"], default="paper_l5")
    # Paper ablations: Table 8 (distribution of w_l) and Fig. 4b (V_i = span{e_i}).
    parser.add_argument("--level-weights", choices=["increasing", "uniform", "decreasing"], default="increasing")
    parser.add_argument("--subspace", choices=["composed", "single"], default="composed")
    parser.add_argument("--tta", choices=["none", "flip"], default="none",
                        help="inference views; 'flip' averages projection energies over the image and its mirror")
    parser.add_argument("--load-checkpoint", default="", help="start from this checkpoint (use --epochs 0 to only predict)")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="", help="e.g. cuda:1; default: cuda if available")
    parser.add_argument("--run-name", default="hier_cos")
    parser.add_argument("--output-dir", default="outputs/predictions")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    hierarchy = build_hierarchy(args.hierarchy)
    node_subspaces = node_subspace_dims(hierarchy, args.subspace)

    train_set = maybe_subset(Cifar100Train(args.data_root, "train", augment=True), args.train_limit, args.seed)
    val_set = Cifar100Train(args.data_root, "val", augment=False)
    test_set = Cifar100TestImages(args.data_root)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

    model = HierCOS(
        num_nodes=hierarchy.num_nodes,
        leaf_subspaces=[node_subspaces[i] for i in hierarchy.leaf_node_indices],
        coarse_subspaces=[node_subspaces[i] for i in hierarchy.level_node_indices[0]],
        backbone=args.backbone,
    ).to(device)
    if args.load_checkpoint:
        state = torch.load(args.load_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
    opt = make_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    resume_path = ckpt_dir / f"{args.run_name}_resume.pt"
    history: list[dict] = []
    start_epoch = 0
    if args.epochs > 0 and resume_path.exists():
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        history = resume["history"]
        start_epoch = len(history)
        torch.set_rng_state(resume["cpu_rng"])
        np.random.set_state(resume["numpy_rng"])
        random.setstate(resume["random_rng"])
        print(f"resumed {args.run_name} at epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for images, fine, _coarse in train_loader:
            images = images.to(device)
            fine = fine.to(device)
            out = model(images)
            loss = hiercos_loss(
                out["coords"],
                fine,
                hierarchy.fine_to_path,
                hierarchy.fine_to_level_labels,
                hierarchy.level_node_indices,
                hierarchy.num_nodes,
                args.alpha,
                level_weights=args.level_weights,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += loss.item() * fine.numel()
            total += fine.numel()
            correct += (out["leaf_scores"].argmax(dim=1) == fine).sum().item()
        scheduler.step()
        history.append({"epoch": epoch + 1, "loss": total_loss / total, "train_acc": correct / total})
        print(f"epoch={epoch+1} loss={total_loss/total:.4f} train_acc={correct/total:.4f}", flush=True)
        if (epoch + 1) % 10 == 0 and epoch + 1 < args.epochs:
            torch.save({
                "model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": scheduler.state_dict(),
                "history": history, "cpu_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                "random_rng": random.getstate(),
            }, resume_path.with_suffix(".tmp"))
            resume_path.with_suffix(".tmp").replace(resume_path)

    if args.epochs > 0:
        torch.save({"model": model.state_dict(), "args": vars(args), "history": history}, ckpt_dir / f"{args.run_name}.pt")
        (ckpt_dir / f"{args.run_name}_history.json").write_text(json.dumps(history, indent=2))
        resume_path.unlink(missing_ok=True)

    level_masks = [mask.to(device) for mask in level_subspace_masks(hierarchy, node_subspaces)]
    val_leaf, val_levels = predict(model, val_loader, device, level_masks, args.tta)
    test_leaf, test_levels = predict(model, test_loader, device, level_masks, args.tta)
    out_path = save_predictions(Path(args.output_dir) / f"{args.run_name}.npz", val_leaf, val_levels, test_leaf, test_levels)
    val_acc = float((val_leaf.argmax(1) == val_set.fine_labels.numpy()).mean())
    print(f"saved {out_path} val_acc={val_acc:.4f}", flush=True)


@torch.no_grad()
def predict(model: HierCOS, loader: DataLoader, device: torch.device, level_masks: list[torch.Tensor], tta: str = "none"):
    """Projection-norm scores for leaves and for every non-leaf taxonomy level."""
    model.eval()
    leaf_scores = []
    level_scores = [[] for _ in level_masks]
    for images, *_ in loader:
        images = images.to(device)
        views = [images] if tta == "none" else [images, images.flip(-1)]
        energy = sum(model(view)["coords"].square() for view in views) / len(views)
        leaf_scores.append(torch.sqrt((energy @ model.leaf_mask.t()).clamp_min(1e-12)).cpu().numpy())
        for idx, mask in enumerate(level_masks):
            level_scores[idx].append(torch.sqrt((energy @ mask.t()).clamp_min(1e-12)).cpu().numpy())
    return np.concatenate(leaf_scores), [np.concatenate(xs) for xs in level_scores]


def level_subspace_masks(hierarchy: Cifar100Hierarchy, node_subspaces: list[list[int]]) -> list[torch.Tensor]:
    """One (K_l, num_nodes) subspace mask per non-leaf level, columns in node order."""
    return [subspace_mask(hierarchy.num_nodes, [node_subspaces[n] for n in hierarchy.level_node_indices[level]])
            for level in range(NON_LEAF_LEVELS)]


def subspace_mask(num_nodes: int, subspaces: list[list[int]]) -> torch.Tensor:
    mask = torch.zeros(len(subspaces), num_nodes)
    for row, dims in enumerate(subspaces):
        mask[row, dims] = 1.0
    return mask


def hierarchy_composed_subspaces(fine_to_path: list[list[int]], num_nodes: int) -> list[list[int]]:
    """Hier-COS: V_i = span(E_i^a U {e_i} U E_i^d)."""
    ancestors: dict[int, set[int]] = {node: set() for node in range(num_nodes)}
    descendants: dict[int, set[int]] = {node: set() for node in range(num_nodes)}
    for path in fine_to_path:
        for level, node in enumerate(path):
            ancestors[node].update(path[:level])
            descendants[node].update(path[level + 1 :])
    return [sorted(ancestors[node] | {node} | descendants[node]) for node in range(num_nodes)]


def node_subspace_dims(hierarchy: Cifar100Hierarchy, mode: str) -> list[list[int]]:
    if mode == "composed":
        return hierarchy_composed_subspaces(hierarchy.fine_to_path, hierarchy.num_nodes)
    if mode == "single":  # ablation of Fig. 4b: no extra dimensions, V_i = span{e_i}
        return [[node] for node in range(hierarchy.num_nodes)]
    raise ValueError(f"unknown subspace mode: {mode}")


def make_optimizer(model: HierCOS, name: str, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name != "sgd":
        raise ValueError(f"unknown optimizer: {name}")

    coord_backbone = getattr(model, "coord_backbone", None)
    if coord_backbone is not None and hasattr(coord_backbone, "head_parameters"):
        return torch.optim.SGD(
            [
                {"params": list(coord_backbone.head_parameters()), "lr": lr},
                {"params": list(coord_backbone.feature_parameters()), "lr": lr * 0.1},
            ],
            momentum=0.9,
            weight_decay=weight_decay,
        )
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)


if __name__ == "__main__":
    main()
