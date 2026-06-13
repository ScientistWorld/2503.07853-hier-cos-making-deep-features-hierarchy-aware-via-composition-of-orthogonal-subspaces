"""Hier-COS model components.

This implements the paper's core computation: features are mapped into a
coordinate system with one fixed orthonormal basis vector per hierarchy node.
Each class is represented by the subspace spanned by its ancestors, itself, and
descendants; leaf logits are projection norms onto leaf subspaces.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SmallCifarBackbone(nn.Module):
    def __init__(self, feature_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WideBasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.use_projection = in_channels != out_channels or stride != 1
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            if self.use_projection
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(x), inplace=True)
        shortcut = self.shortcut(out) if self.use_projection else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return out + shortcut


class WideResNetBackbone(nn.Module):
    """Compact CIFAR WideResNet backbone, matching the paper's backbone family."""

    def __init__(self, depth: int = 16, widen_factor: int = 4, feature_dim: int = 256) -> None:
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError("WideResNet depth must satisfy depth = 6n + 4")
        blocks_per_group = (depth - 4) // 6
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.conv1 = nn.Conv2d(3, channels[0], 3, padding=1, bias=False)
        self.group1 = self._make_group(channels[0], channels[1], blocks_per_group, stride=1)
        self.group2 = self._make_group(channels[1], channels[2], blocks_per_group, stride=2)
        self.group3 = self._make_group(channels[2], channels[3], blocks_per_group, stride=2)
        self.bn = nn.BatchNorm2d(channels[3])
        self.proj = nn.Sequential(
            nn.Linear(channels[3], feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _make_group(in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [WideBasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(WideBasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.group1(out)
        out = self.group2(out)
        out = self.group3(out)
        out = F.relu(self.bn(out), inplace=True)
        out = F.avg_pool2d(out, out.shape[-1])
        return self.proj(out.flatten(1))


def make_backbone(name: str, feature_dim: int = 256) -> nn.Module:
    if name == "small":
        return SmallCifarBackbone(feature_dim=feature_dim)
    if name == "wrn16_4":
        return WideResNetBackbone(depth=16, widen_factor=4, feature_dim=feature_dim)
    raise ValueError(f"unknown backbone: {name}")


class Transformation(nn.Module):
    """Five-layer HAFrame-style transform into the Hier-COS node space."""

    def __init__(self, in_dim: int, num_nodes: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = in_dim
        for _ in range(4):
            layers.extend([nn.Linear(current, num_nodes), nn.BatchNorm1d(num_nodes), nn.PReLU(num_nodes)])
            current = num_nodes
        layers.append(nn.Linear(current, num_nodes, bias=False))
        self.layers = nn.Sequential(*layers)
        final = self.layers[-1]
        assert isinstance(final, nn.Linear)
        with torch.no_grad():
            final.weight.copy_(torch.eye(num_nodes))
        final.weight.requires_grad_(False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.layers(z)


class HierCOS(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        leaf_subspaces: list[list[int]],
        coarse_subspaces: list[list[int]],
        backbone: str = "small",
    ) -> None:
        super().__init__()
        self.backbone = make_backbone(backbone, feature_dim=256)
        self.transform = Transformation(256, num_nodes)
        self.register_buffer("leaf_mask", _subspace_mask(num_nodes, leaf_subspaces))
        self.register_buffer("coarse_mask", _subspace_mask(num_nodes, coarse_subspaces))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        coords = self.transform(self.backbone(x))
        squared = coords.square()
        leaf_scores = torch.sqrt(torch.matmul(squared, self.leaf_mask.t()).clamp_min(1e-12))
        coarse_scores = torch.sqrt(torch.matmul(squared, self.coarse_mask.t()).clamp_min(1e-12))
        return {"coords": coords, "leaf_scores": leaf_scores, "coarse_scores": coarse_scores}


class FlatCE(nn.Module):
    def __init__(self, num_classes: int = 100, backbone: str = "small") -> None:
        super().__init__()
        self.backbone = make_backbone(backbone, feature_dim=256)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


def _subspace_mask(num_nodes: int, subspaces: list[list[int]]) -> torch.Tensor:
    mask = torch.zeros(len(subspaces), num_nodes)
    for row, dims in enumerate(subspaces):
        mask[row, dims] = 1.0
    return mask


def path_targets(fine_labels: torch.Tensor, fine_to_path: list[list[int]], num_nodes: int) -> torch.Tensor:
    """Paper's weighted one-hot path distribution P."""
    out = torch.zeros(fine_labels.numel(), num_nodes, device=fine_labels.device)
    path_table = torch.tensor(fine_to_path, device=fine_labels.device, dtype=torch.long)
    paths = path_table[fine_labels]
    height = paths.shape[1]
    weights = torch.exp(
        1.0 / torch.arange(height, 0, -1, device=fine_labels.device, dtype=torch.float32)
    )
    weights = weights / weights.sum()
    rows = torch.arange(fine_labels.numel(), device=fine_labels.device).unsqueeze(1).expand_as(paths)
    out[rows, paths] = weights.unsqueeze(0).expand_as(paths).float()
    return out


def hiercos_loss(
    coords: torch.Tensor,
    fine_labels: torch.Tensor,
    fine_to_path: list[list[int]],
    num_nodes: int,
    alpha: float,
) -> torch.Tensor:
    targets = path_targets(fine_labels, fine_to_path, num_nodes)
    log_probs = F.log_softmax(coords.abs(), dim=1)
    kl = F.kl_div(log_probs, targets, reduction="batchmean")

    path_table = torch.tensor(fine_to_path, device=fine_labels.device, dtype=torch.long)
    path_nodes = path_table[fine_labels]
    active = torch.zeros_like(coords)
    active.scatter_(1, path_nodes, 1.0)
    normalized = F.normalize(coords, p=2, dim=1, eps=1e-8)
    reg = (active - normalized).abs().sum(dim=1).mean()
    return kl + alpha * reg
