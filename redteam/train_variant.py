"""Audit training runs; uses the supplied method, loss, data and preprocessing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.cifar100_dataset import Cifar100Pickle, maybe_subset
from data.cifar100_hierarchy import build_hierarchy
from method.hiercos_model import HierCOS, hiercos_loss
from method.train_hiercos import hierarchy_composed_subspaces, make_optimizer, predict


def save_predictions(model, loader, device, hierarchy, directory):
    directory.mkdir(parents=True, exist_ok=True)
    labels, coarse, leaves, coarse_scores, level_labels, levels = predict(model, loader, device, hierarchy)
    np.savez_compressed(
        directory / "hier_cos.npz", method_name="hier_cos", hierarchy=hierarchy.name,
        labels=labels, coarse_labels=coarse, leaf_scores=leaves,
        coarse_scores=coarse_scores, level_labels=level_labels,
        **{f"level_scores_{i}": x for i, x in enumerate(levels)},
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["seed", "extended", "bn", "checkpoint"])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)
    checkpoint = torch.load(ROOT / "checkpoints/hier_cos.pt", map_location="cpu", weights_only=False)
    cfg = checkpoint["args"].copy()
    cfg["seed"] = args.seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    h = build_hierarchy(cfg["hierarchy"])
    subspaces = hierarchy_composed_subspaces(h.fine_to_path, h.num_nodes)
    model = HierCOS(h.num_nodes, [subspaces[i] for i in h.leaf_node_indices],
                    [subspaces[i] for i in h.level_node_indices[0]], cfg["backbone"]).to(device)
    train = maybe_subset(Cifar100Pickle(ROOT / cfg["data_root"], "train", augment=args.mode != "bn"),
                         cfg["train_limit"], args.seed)
    test = Cifar100Pickle(ROOT / cfg["data_root"], "test", augment=False)
    train_loader = DataLoader(train, batch_size=cfg["batch_size"], shuffle=args.mode != "bn",
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)
    case = {"seed": f"reference_seed_{args.seed}", "extended": "variant_more_training",
            "bn": "variant_bn_recalibration", "checkpoint": "reference_checkpoint"}[args.mode]
    outdir = ROOT / "redteam" / case
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))
    t0 = time.monotonic()
    if args.mode != "seed":
        model.load_state_dict(checkpoint["model"])
    del checkpoint
    if args.mode == "bn":
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.reset_running_stats()
                module.momentum = None
        with torch.no_grad():
            for images, _, _ in train_loader:
                model(images.to(device))
        history = [{"operation": "recompute BN running moments on CIFAR training images only"}]
    elif args.mode == "checkpoint":
        history = []
    else:
        epochs = cfg["epochs"] if args.mode == "seed" else 25
        lr = cfg["lr"] if args.mode == "seed" else cfg["lr"] * 0.05
        opt = make_optimizer(model, cfg["optimizer"], lr, cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        history = []
        for epoch in range(epochs):
            model.train()
            total = correct = 0
            loss_sum = 0.
            epoch_t0 = time.monotonic()
            for images, fine, _ in train_loader:
                images, fine = images.to(device), fine.to(device)
                output = model(images)
                loss = hiercos_loss(output["coords"], fine, h.fine_to_path, h.fine_to_level_labels,
                                    h.level_node_indices, h.num_nodes, cfg["alpha"])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += fine.numel()
                correct += int((output["leaf_scores"].argmax(1) == fine).sum())
                loss_sum += loss.item() * fine.numel()
            scheduler.step()
            row = {"epoch": epoch + 1, "loss": loss_sum / total,
                   "train_accuracy": correct / total, "seconds": time.monotonic() - epoch_t0}
            history.append(row)
            print(json.dumps({"case": case, **row}), flush=True)
            (outdir / "history.json").write_text(json.dumps(history, indent=2))
            if args.mode == "seed" and args.seed == 7 and epoch == 0:
                rng_state = (torch.get_rng_state(), torch.cuda.get_rng_state(device),
                             np.random.get_state(), random.getstate())
                save_predictions(model, test_loader, device, h,
                                 ROOT / "redteam/variant_less_training/artifacts")
                (ROOT / "redteam/variant_less_training/config.json").write_text(
                    json.dumps({**cfg, "stop_after_epoch": 1, "scheduler_T_max": epochs}, indent=2))
                torch.set_rng_state(rng_state[0])
                torch.cuda.set_rng_state(rng_state[1], device)
                np.random.set_state(rng_state[2])
                random.setstate(rng_state[3])
        cfg["additional_epochs"] = 25 if args.mode == "extended" else 0
    save_predictions(model, test_loader, device, h, outdir / "artifacts")
    torch.save({"model": model.state_dict(), "args": cfg}, outdir / "artifacts/model.pt")
    (outdir / "training_summary.json").write_text(json.dumps({
        "mode": args.mode, "seed": args.seed, "seconds": time.monotonic() - t0,
        "epochs_completed": len(history), "gpu": args.gpu}, indent=2))


if __name__ == "__main__":
    main()
