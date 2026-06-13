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


class OfficialWideBasicBlock(nn.Module):
    """WideResNet block matching the released Hier-COS CIFAR backbone."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        activate_before_residual: bool = False,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.equal_channels = in_channels == out_channels
        self.activate_before_residual = activate_before_residual
        self.dropout_rate = dropout_rate
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=0.001)
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.001)
        self.relu2 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.shortcut = (
            nn.Identity()
            if self.equal_channels and stride == 1
            else nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.equal_channels and self.activate_before_residual:
            x = self.relu1(self.bn1(x))
            out = x
        else:
            out = self.relu1(self.bn1(x))
        out = self.conv1(out if self.equal_channels else x)
        out = self.relu2(self.bn2(out))
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.conv2(out)
        return self.shortcut(x) + out


class OfficialNetworkBlock(nn.Module):
    def __init__(
        self,
        layers: int,
        in_channels: int,
        out_channels: int,
        stride: int,
        activate_before_residual: bool = False,
    ) -> None:
        super().__init__()
        blocks = []
        for idx in range(layers):
            blocks.append(
                OfficialWideBasicBlock(
                    in_channels if idx == 0 else out_channels,
                    out_channels,
                    stride if idx == 0 else 1,
                    activate_before_residual=activate_before_residual and idx == 0,
                )
            )
        self.layer = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class OfficialWideResNetFeatures(nn.Module):
    """WRN-28-8 feature extractor used by the official CIFAR experiments."""

    def __init__(self, depth: int = 28, widen_factor: int = 8) -> None:
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError("WideResNet depth must satisfy depth = 6n + 4")
        layers_per_group = (depth - 4) // 6
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.out_dim = channels[3]
        self.conv1 = nn.Conv2d(3, channels[0], 3, stride=1, padding=1, bias=False)
        self.block1 = OfficialNetworkBlock(
            layers_per_group, channels[0], channels[1], 1, activate_before_residual=True
        )
        self.block2 = OfficialNetworkBlock(layers_per_group, channels[1], channels[2], 2)
        self.block3 = OfficialNetworkBlock(layers_per_group, channels[2], channels[3], 2)
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self._init_weights()

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        return self.relu(self.bn1(out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._features(x)
        out = F.adaptive_avg_pool2d(out, 1)
        return out.flatten(1)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.weight.data.fill_(1.0)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()


class ResidualPointLayer(nn.Module):
    """Residual point-wise transformation block from the released HAFrame head."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.shortcut = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.PReLU(out_dim),
            nn.Linear(out_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.PReLU(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.net(x)


class NarrowResidualTransformationHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            ResidualPointLayer(in_dim, out_dim),
            ResidualPointLayer(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class OfficialHAFrameWideResNet(nn.Module):
    """Official-style CIFAR HAFrame-WRN coordinate backbone for Hier-COS."""

    def __init__(self, num_nodes: int, depth: int = 28, widen_factor: int = 8) -> None:
        super().__init__()
        self.features = OfficialWideResNetFeatures(depth=depth, widen_factor=widen_factor)
        self.downsample = nn.Conv2d(self.features.out_dim, num_nodes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(num_nodes, momentum=0.001)
        self.projection = NarrowResidualTransformationHead(num_nodes, num_nodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features._features(x)
        out = self.downsample(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        out = self.bn2(out)
        return self.projection(out)

    def feature_parameters(self):
        yield from self.features.parameters()

    def head_parameters(self):
        yield from self.downsample.parameters()
        yield from self.bn2.parameters()
        yield from self.projection.parameters()


def make_backbone(name: str, feature_dim: int = 256) -> nn.Module:
    if name == "small":
        model = SmallCifarBackbone(feature_dim=feature_dim)
        model.out_dim = feature_dim  # type: ignore[attr-defined]
        return model
    if name == "wrn16_4":
        model = WideResNetBackbone(depth=16, widen_factor=4, feature_dim=feature_dim)
        model.out_dim = feature_dim  # type: ignore[attr-defined]
        return model
    if name == "official_wrn28_8":
        return OfficialWideResNetFeatures(depth=28, widen_factor=8)
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
        self.coord_backbone: nn.Module | None = None
        if backbone == "official_wrn28_8":
            self.coord_backbone = OfficialHAFrameWideResNet(num_nodes=num_nodes, depth=28, widen_factor=8)
            self.backbone = None
            self.transform = None
        else:
            self.backbone = make_backbone(backbone, feature_dim=256)
            self.transform = Transformation(256, num_nodes)
        self.register_buffer("leaf_mask", _subspace_mask(num_nodes, leaf_subspaces))
        self.register_buffer("coarse_mask", _subspace_mask(num_nodes, coarse_subspaces))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.coord_backbone is not None:
            coords = self.coord_backbone(x)
        else:
            assert self.backbone is not None and self.transform is not None
            coords = self.transform(self.backbone(x))
        squared = coords.square()
        leaf_scores = torch.sqrt(torch.matmul(squared, self.leaf_mask.t()).clamp_min(1e-12))
        coarse_scores = torch.sqrt(torch.matmul(squared, self.coarse_mask.t()).clamp_min(1e-12))
        return {"coords": coords, "leaf_scores": leaf_scores, "coarse_scores": coarse_scores}


class FlatCE(nn.Module):
    def __init__(self, num_classes: int = 100, backbone: str = "small") -> None:
        super().__init__()
        self.backbone = make_backbone(backbone, feature_dim=256)
        self.classifier = nn.Linear(int(getattr(self.backbone, "out_dim", 256)), num_classes)

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
    # The released Hier-COS code stores P as squared L2-normalized path weights,
    # which sums to one and gives the finest levels the intended larger mass.
    weights = (weights / weights.norm(p=2)).square()
    rows = torch.arange(fine_labels.numel(), device=fine_labels.device).unsqueeze(1).expand_as(paths)
    out[rows, paths] = weights.unsqueeze(0).expand_as(paths).float()
    return out


def hiercos_loss(
    coords: torch.Tensor,
    fine_labels: torch.Tensor,
    fine_to_path: list[list[int]],
    fine_to_level_labels: list[list[int]],
    level_node_indices: list[list[int]],
    num_nodes: int,
    alpha: float,
) -> torch.Tensor:
    targets = path_targets(fine_labels, fine_to_path, num_nodes)
    log_probs = F.log_softmax(coords.abs(), dim=1)
    kl = F.kl_div(log_probs, targets, reduction="batchmean")

    label_table = torch.tensor(fine_to_level_labels, device=fine_labels.device, dtype=torch.long)
    level_labels = label_table[fine_labels]
    reg = coords.new_tensor(0.0)
    for level, nodes in enumerate(level_node_indices):
        node_tensor = torch.tensor(nodes, device=fine_labels.device, dtype=torch.long)
        level_coords = coords[:, node_tensor].abs()
        valid = level_labels[:, level] >= 0
        if valid.any():
            normalized = F.normalize(level_coords[valid], p=2, dim=1, eps=1e-8)
            onehot = F.one_hot(level_labels[valid, level], num_classes=len(nodes)).float()
            reg = reg + (onehot - normalized).abs().sum(dim=1).mean()
        if (~valid).any():
            reg = reg + level_coords[~valid].sum(dim=1).mean()
    return kl + alpha * reg
