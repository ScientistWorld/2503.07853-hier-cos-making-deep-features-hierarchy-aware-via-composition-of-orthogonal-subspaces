"""Baseline CIFAR classifiers used for method-agnostic comparisons.

This module intentionally does not import from ``method/``.  The CE baseline
shares the same backbone options as the Hier-COS runner, but remains a
standalone baseline that future method implementations can compare against.
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
        self.out_dim = feature_dim

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
        self.out_dim = feature_dim

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
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


def make_backbone(name: str, feature_dim: int = 256) -> nn.Module:
    if name == "small":
        return SmallCifarBackbone(feature_dim=feature_dim)
    if name == "wrn16_4":
        return WideResNetBackbone(depth=16, widen_factor=4, feature_dim=feature_dim)
    if name == "official_wrn28_8":
        return OfficialWideResNetFeatures(depth=28, widen_factor=8)
    raise ValueError(f"unknown backbone: {name}")


class FlatCE(nn.Module):
    def __init__(self, num_classes: int = 100, backbone: str = "small") -> None:
        super().__init__()
        self.backbone = make_backbone(backbone, feature_dim=256)
        self.classifier = nn.Linear(int(getattr(self.backbone, "out_dim", 256)), num_classes)
        nn.init.xavier_normal_(self.classifier.weight)
        self.classifier.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))
