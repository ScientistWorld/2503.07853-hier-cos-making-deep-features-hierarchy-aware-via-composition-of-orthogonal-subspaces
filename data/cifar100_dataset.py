"""CIFAR-100 loaders for the benchmark protocol (independent of torchvision).

Protocol (paper Table 5): 45,000 training images, a fixed 5,000-image
validation split carved from the official training set
(`data/cifar100_val_split.json`), and the 10,000 official test images. Test
labels are not available in `data/`; the test loader yields images only, in
official test order.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


MEAN = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
STD = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)
SPLIT_FILE = Path(__file__).resolve().parent / "cifar100_val_split.json"
UNLABELLED = -1


def cifar_root(path: str | Path = "data/downloads/cifar-100-python") -> Path:
    root = Path(path)
    if (root / "train").exists():
        return root
    nested = root / "cifar-100-python"
    if (nested / "train").exists():
        return nested
    raise FileNotFoundError(f"Could not find CIFAR-100 python files under {root}; run scripts/download.sh")


def val_indices() -> list[int]:
    return list(json.loads(SPLIT_FILE.read_text())["val_indices"])


def train_indices() -> list[int]:
    held_out = set(val_indices())
    return [idx for idx in range(50_000) if idx not in held_out]


def _augment(image: torch.Tensor) -> torch.Tensor:
    padded = torch.nn.functional.pad(image.unsqueeze(0), (4, 4, 4, 4), mode="reflect").squeeze(0)
    top = torch.randint(0, 9, ()).item()
    left = torch.randint(0, 9, ()).item()
    cropped = padded[:, top:top + 32, left:left + 32]
    if torch.rand(()) < 0.5:
        cropped = torch.flip(cropped, dims=(2,))
    return cropped


class Cifar100Train(Dataset):
    """Labelled images from the official training pickle.

    split: "train" (45K, the complement of the validation split), "val" (the
    fixed 5K validation split, in `val_indices()` order) or "all" (50K).
    """

    def __init__(self, root: str | Path, split: str = "train", augment: bool = False) -> None:
        if split not in {"train", "val", "all"}:
            raise ValueError("split must be 'train', 'val' or 'all'")
        self.root = cifar_root(root)
        self.augment = augment
        with (self.root / "train").open("rb") as f:
            obj = pickle.load(f, encoding="latin1")
        images = obj["data"].reshape(-1, 3, 32, 32)
        fine = np.asarray(obj["fine_labels"], dtype=np.int64)
        coarse = np.asarray(obj["coarse_labels"], dtype=np.int64)
        if split == "train":
            rows = np.asarray(train_indices(), dtype=np.int64)
        elif split == "val":
            rows = np.asarray(val_indices(), dtype=np.int64)
        else:
            rows = np.arange(len(fine), dtype=np.int64)
        self.indices = rows
        self.images = torch.from_numpy(images[rows].astype("float32") / 255.0)
        self.fine_labels = torch.from_numpy(fine[rows])
        self.coarse_labels = torch.from_numpy(coarse[rows])

    def __len__(self) -> int:
        return int(self.fine_labels.numel())

    def __getitem__(self, idx: int):
        image = self.images[idx]
        if self.augment:
            image = _augment(image)
        image = (image - MEAN) / STD
        return image, self.fine_labels[idx], self.coarse_labels[idx]


class Cifar100TestImages(Dataset):
    """The 10,000 official test images, unlabelled, in official test order."""

    def __init__(self, root: str | Path) -> None:
        self.root = cifar_root(root)
        raw = np.fromfile(self.root / "test_images.u8", dtype=np.uint8)
        if raw.size != 10_000 * 3 * 32 * 32:
            raise ValueError("test_images.u8 has unexpected size; rerun scripts/download.sh")
        self.images = torch.from_numpy(raw.reshape(-1, 3, 32, 32).astype("float32") / 255.0)

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int):
        image = (self.images[idx] - MEAN) / STD
        return image, UNLABELLED, UNLABELLED


def maybe_subset(dataset: Dataset, limit: int | None, seed: int) -> Dataset:
    """Random training subset for quick experiments (never use for val/test)."""
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=limit, replace=False)
    return Subset(dataset, indices.tolist())
