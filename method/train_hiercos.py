from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.cifar100_dataset import Cifar100Pickle, maybe_subset
from data.cifar100_hierarchy import build_hierarchy
from method.hiercos_model import HierCOS, hiercos_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/downloads/cifar-100-python")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=5e-2)
    parser.add_argument("--backbone", choices=["small", "wrn16_4"], default="small")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--run-name", default="hier_cos")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hierarchy = build_hierarchy()

    train_set = maybe_subset(Cifar100Pickle(args.data_root, "train", augment=True), args.train_limit, args.seed)
    test_set = maybe_subset(Cifar100Pickle(args.data_root, "test", augment=False), args.test_limit, args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = HierCOS(
        num_nodes=hierarchy.num_nodes,
        leaf_subspaces=hierarchy.leaf_to_subspace,
        coarse_subspaces=[hierarchy.node_to_subspace[i] for i in hierarchy.coarse_node_indices],
        backbone=args.backbone,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for images, fine, _coarse in train_loader:
            images = images.to(device)
            fine = fine.to(device)
            out = model(images)
            loss = hiercos_loss(out["coords"], fine, hierarchy.fine_to_coarse, hierarchy.num_nodes, args.alpha)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += loss.item() * fine.numel()
            total += fine.numel()
            correct += (out["leaf_scores"].argmax(dim=1) == fine).sum().item()
        scheduler.step()
        history.append({"epoch": epoch + 1, "loss": total_loss / total, "train_acc": correct / total})
        print(f"epoch={epoch+1} loss={total_loss/total:.4f} train_acc={correct/total:.4f}", flush=True)

    pred_dir = Path("outputs/predictions")
    ckpt_dir = Path("checkpoints")
    pred_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    labels, coarse_labels, leaf_scores, coarse_scores = predict(model, test_loader, device)
    np.savez_compressed(
        pred_dir / f"{args.run_name}.npz",
        method_name=args.run_name,
        labels=labels,
        coarse_labels=coarse_labels,
        leaf_scores=leaf_scores,
        coarse_scores=coarse_scores,
    )
    torch.save({"model": model.state_dict(), "args": vars(args), "history": history}, ckpt_dir / f"{args.run_name}.pt")
    (ckpt_dir / f"{args.run_name}_history.json").write_text(json.dumps(history, indent=2))


@torch.no_grad()
def predict(model: HierCOS, loader: DataLoader, device: torch.device):
    model.eval()
    labels, coarse_labels, leaf_scores, coarse_scores = [], [], [], []
    for images, fine, coarse in loader:
        out = model(images.to(device))
        labels.append(fine.numpy())
        coarse_labels.append(coarse.numpy())
        leaf_scores.append(out["leaf_scores"].cpu().numpy())
        coarse_scores.append(out["coarse_scores"].cpu().numpy())
    return map(lambda xs: np.concatenate(xs, axis=0), (labels, coarse_labels, leaf_scores, coarse_scores))


if __name__ == "__main__":
    main()
