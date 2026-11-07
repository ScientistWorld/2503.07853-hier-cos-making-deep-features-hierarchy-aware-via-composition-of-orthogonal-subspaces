"""Flat cross-entropy baseline (paper's "Cross Entropy" row).

Multi-level scores follow the paper: a superclass's probability is the sum of
its leaves' softmax probabilities. Writes the standard prediction artifact
(default ``outputs/baseline/cross_entropy.npz``). The benchmark's scored
cross-entropy row comes from the evaluator-owned frozen copy of this artifact
(see ``baseline/freeze_fixed_baseline.py``), not from whatever is on disk.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.cifar100_dataset import Cifar100TestImages, Cifar100Train, maybe_subset
from data.cifar100_hierarchy import build_hierarchy
from data.prediction_format import save_predictions
from baseline.models import FlatCE

NON_LEAF_LEVELS = 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/downloads/cifar-100-python")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--backbone", choices=["small", "wrn16_4", "official_wrn28_8"], default="official_wrn28_8")
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="sgd")
    parser.add_argument("--hierarchy", choices=["public", "paper_l5"], default="paper_l5")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--select", choices=["best_val", "last"], default="best_val",
                        help="official pipeline: validate every epoch, keep the best validation top-1 checkpoint")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="", help="e.g. cuda:1; default: cuda if available")
    parser.add_argument("--run-name", default="cross_entropy")
    parser.add_argument("--output-dir", default="outputs/baseline")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    hierarchy = build_hierarchy(args.hierarchy)
    train_set = maybe_subset(Cifar100Train(args.data_root, "train", augment=True), args.train_limit, args.seed)
    val_set = Cifar100Train(args.data_root, "val", augment=False)
    test_set = Cifar100TestImages(args.data_root)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

    model = FlatCE(num_classes=100, backbone=args.backbone).to(device)
    opt = make_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    Path("checkpoints").mkdir(exist_ok=True)
    best_path = Path("checkpoints") / f"{args.run_name}_best_state.pt"
    best_acc, best_epoch = -1.0, 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for images, fine, _coarse in train_loader:
            images = images.to(device)
            fine = fine.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, fine)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += loss.item() * fine.numel()
            total += fine.numel()
            correct += (logits.argmax(dim=1) == fine).sum().item()
        scheduler.step()
        val_acc = float("nan")
        if args.select == "best_val":
            val_acc = val_top1(model, val_loader, device)
            if val_acc > best_acc:
                best_acc, best_epoch = val_acc, epoch + 1
                torch.save(model.state_dict(), best_path)
        print(f"epoch={epoch+1} loss={total_loss/total:.4f} train_acc={correct/total:.4f} val_acc={val_acc:.4f}", flush=True)
    if args.select == "best_val" and best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False))
        best_path.unlink()
        print(f"selected epoch {best_epoch} (val top-1 {best_acc:.4f})", flush=True)

    val_leaf = predict(model, val_loader, device)
    test_leaf = predict(model, test_loader, device)
    out_path = save_predictions(
        Path(args.output_dir) / f"{args.run_name}.npz",
        val_leaf, level_scores_from_leaves(val_leaf, hierarchy),
        test_leaf, level_scores_from_leaves(test_leaf, hierarchy),
    )
    val_acc = float((val_leaf.argmax(1) == val_set.fine_labels.numpy()).mean())
    print(f"saved {out_path} val_acc={val_acc:.4f}", flush=True)
    torch.save({"model": model.state_dict(), "args": vars(args), "selected_epoch": best_epoch or args.epochs},
               Path("checkpoints") / f"{args.run_name}.pt")


def level_scores_from_leaves(logits: np.ndarray, hierarchy) -> list[np.ndarray]:
    """Superclass probability = sum of its leaves' softmax probabilities."""
    probs = softmax_np(logits)
    out = []
    for level in range(NON_LEAF_LEVELS):
        leaf_sets = hierarchy.level_to_leaf_sets[level]
        scores = np.zeros((logits.shape[0], len(leaf_sets)), dtype=np.float32)
        for node_idx, leaves in enumerate(leaf_sets):
            scores[:, node_idx] = probs[:, leaves].sum(axis=1)
        out.append(scores)
    return out


@torch.no_grad()
def val_top1(model: FlatCE, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for images, fine, _coarse in loader:
        correct += int((model(images.to(device)).argmax(dim=1).cpu() == fine).sum())
        total += int(fine.numel())
    model.train()
    return correct / total


@torch.no_grad()
def predict(model: FlatCE, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    return np.concatenate([model(images.to(device)).cpu().numpy() for images, *_ in loader])


def softmax_np(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def make_optimizer(model: FlatCE, name: str, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name != "sgd":
        raise ValueError(f"unknown optimizer: {name}")
    if getattr(model.backbone, "out_dim", 256) != 256:
        return torch.optim.SGD(
            [
                {"params": model.classifier.parameters(), "lr": lr},
                {"params": model.backbone.parameters(), "lr": lr * 0.1},
            ],
            momentum=0.9,
            weight_decay=weight_decay,
        )
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)


if __name__ == "__main__":
    main()
