from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.cifar100_dataset import Cifar100Pickle, maybe_subset
from data.cifar100_hierarchy import build_hierarchy
from method.hiercos_model import FlatCE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/downloads/cifar-100-python")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--run-name", default="baseline_ce")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hierarchy = build_hierarchy()
    train_set = maybe_subset(Cifar100Pickle(args.data_root, "train", augment=True), args.train_limit, args.seed)
    test_set = maybe_subset(Cifar100Pickle(args.data_root, "test", augment=False), args.test_limit, args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = FlatCE(num_classes=100).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
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
        print(f"epoch={epoch+1} loss={total_loss/total:.4f} train_acc={correct/total:.4f}", flush=True)

    labels, coarse_labels, leaf_scores = predict(model, test_loader, device)
    coarse_scores = np.zeros((leaf_scores.shape[0], hierarchy.num_coarse), dtype=np.float32)
    probs = softmax_np(leaf_scores)
    for fine_idx, coarse_idx in enumerate(hierarchy.fine_to_coarse):
        coarse_scores[:, coarse_idx] += probs[:, fine_idx]
    pred_dir = Path("outputs/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_dir / f"{args.run_name}.npz",
        method_name=args.run_name,
        labels=labels,
        coarse_labels=coarse_labels,
        leaf_scores=leaf_scores,
        coarse_scores=coarse_scores,
    )
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args)}, Path("checkpoints") / f"{args.run_name}.pt")


@torch.no_grad()
def predict(model: FlatCE, loader: DataLoader, device: torch.device):
    model.eval()
    labels, coarse_labels, leaf_scores = [], [], []
    for images, fine, coarse in loader:
        logits = model(images.to(device))
        labels.append(fine.numpy())
        coarse_labels.append(coarse.numpy())
        leaf_scores.append(logits.cpu().numpy())
    return map(lambda xs: np.concatenate(xs, axis=0), (labels, coarse_labels, leaf_scores))


def softmax_np(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    main()
