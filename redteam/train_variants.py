"""Run original full-scale training; save seed, weaker, and longer-trained models.

Only executed as a scheduled GPU job. The training procedure and prediction
writer are imported from the supplied method. No hidden evaluator is imported.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.cifar100_dataset import Cifar100Pickle
from data.cifar100_hierarchy import build_hierarchy
from method.hiercos_model import HierCOS, hiercos_loss
from method.train_hiercos import hierarchy_composed_subspaces, make_optimizer, predict

ROOT = Path(__file__).resolve().parents[1]


def save_prediction(model, loader, device, hierarchy, name, history):
    folder = ROOT / 'redteam' / name
    folder.mkdir(parents=True, exist_ok=True)
    labels, coarse_labels, leaf_scores, coarse_scores, level_labels, levels = predict(
        model, loader, device, hierarchy
    )
    np.savez_compressed(folder / 'prediction.npz', method_name='hier_cos',
        hierarchy='paper_l5', labels=labels, coarse_labels=coarse_labels,
        leaf_scores=leaf_scores, coarse_scores=coarse_scores, level_labels=level_labels,
        **{f'level_scores_{i}': scores for i, scores in enumerate(levels)})
    (folder / 'training.json').write_text(json.dumps(history, indent=2))
    torch.save({'model': model.state_dict(), 'history': history}, folder / 'model.pt')


def train(seed, gpu, extend=False):
    torch.set_num_threads(1)
    torch.cuda.set_device(gpu)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda', gpu)
    hierarchy = build_hierarchy('paper_l5')
    spaces = hierarchy_composed_subspaces(hierarchy.fine_to_path, hierarchy.num_nodes)
    train_loader = DataLoader(Cifar100Pickle('data/downloads/cifar-100-python', 'train', augment=True),
        batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(Cifar100Pickle('data/downloads/cifar-100-python', 'test', augment=False),
        batch_size=64, shuffle=False, num_workers=2)
    model = HierCOS(hierarchy.num_nodes,
        [spaces[i] for i in hierarchy.leaf_node_indices],
        [spaces[i] for i in hierarchy.level_node_indices[0]],
        backbone='official_wrn28_8').to(device)
    optimizer = make_optimizer(model, 'sgd', 0.1, 0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    history = []
    start = time.monotonic()
    for epoch in range(150 if extend else 100):
        if epoch == 100:
            # A genuine extra-training variant: same model, data, loss and
            # augmentation; restart a small learning rate for 50 more epochs.
            optimizer = make_optimizer(model, 'sgd', 0.01, 0.0005)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        model.train()
        total = correct = 0
        total_loss = 0.0
        for images, fine, _ in train_loader:
            images, fine = images.to(device), fine.to(device)
            outputs = model(images)
            loss = hiercos_loss(outputs['coords'], fine, hierarchy.fine_to_path,
                hierarchy.fine_to_level_labels, hierarchy.level_node_indices,
                hierarchy.num_nodes, 0.05)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = fine.numel()
            total += count
            total_loss += loss.item() * count
            correct += (outputs['leaf_scores'].argmax(dim=1) == fine).sum().item()
        scheduler.step()
        row = {'epoch': epoch+1, 'loss': total_loss / total,
            'train_acc': correct / total, 'elapsed_seconds': time.monotonic()-start}
        history.append(row)
        print(json.dumps({'seed': seed, **row}), flush=True)
        # Predicting consumes RNG when creating the DataLoader iterator. Save
        # and restore all training RNG state so these diagnostics do not alter
        # the remaining training trajectory.
        name = None
        if seed == 7 and epoch == 9:
            name = 'variant_weaker'
        elif epoch == 99:
            name = f'seed_{seed}'
        elif epoch == 149:
            name = 'variant_longer_training'
        if name:
            cpu_state = torch.get_rng_state()
            gpu_state = torch.cuda.get_rng_state(device)
            numpy_state = np.random.get_state()
            save_prediction(model, test_loader, device, hierarchy, name, history)
            torch.set_rng_state(cpu_state)
            torch.cuda.set_rng_state(gpu_state, device)
            np.random.set_state(numpy_state)
    config = {'seed':seed, 'epochs':150 if extend else 100, 'batch_size':64,
        'optimizer':'sgd', 'lr':0.1, 'weight_decay':0.0005, 'alpha':0.05,
        'backbone':'official_wrn28_8', 'hierarchy':'paper_l5',
        'train_examples':50000, 'test_examples':10000, 'gpu':gpu,
        'extended_lr':0.01 if extend else None, 'extension_epochs':50 if extend else 0}
    (ROOT / 'redteam' / f'seed_{seed}' / 'config.json').write_text(json.dumps(config,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--extend', action='store_true')
    args = parser.parse_args()
    train(args.seed, args.gpu, args.extend)
