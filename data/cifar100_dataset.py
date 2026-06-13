"""Small CIFAR-100 loader independent of torchvision datasets."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


MEAN = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
STD = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)


def cifar_root(path: str | Path = "data/downloads/cifar-100-python") -> Path:
    root = Path(path)
    if (root / "train").exists():
        return root
    nested = root / "cifar-100-python"
    if (nested / "train").exists():
        return nested
    raise FileNotFoundError(f"Could not find CIFAR-100 python files under {root}")


class Cifar100Pickle(Dataset):
    def __init__(self, root: str | Path, split: str, augment: bool = False) -> None:
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        self.root = cifar_root(root)
        self.split = split
        self.augment = augment
        with (self.root / split).open("rb") as f:
            obj = pickle.load(f, encoding="latin1")
        data = obj["data"].reshape(-1, 3, 32, 32).astype("float32") / 255.0
        self.images = torch.from_numpy(data)
        self.fine_labels = torch.tensor(obj["fine_labels"], dtype=torch.long)
        self.coarse_labels = torch.tensor(obj["coarse_labels"], dtype=torch.long)

    def __len__(self) -> int:
        return int(self.fine_labels.numel())

    def __getitem__(self, idx: int):
        image = self.images[idx]
        if self.augment:
            image = self._augment(image)
        image = (image - MEAN) / STD
        return image, self.fine_labels[idx], self.coarse_labels[idx]

    @staticmethod
    def _augment(image: torch.Tensor) -> torch.Tensor:
        padded = torch.nn.functional.pad(image.unsqueeze(0), (4, 4, 4, 4), mode="reflect").squeeze(0)
        top = torch.randint(0, 9, ()).item()
        left = torch.randint(0, 9, ()).item()
        cropped = padded[:, top:top + 32, left:left + 32]
        if torch.rand(()) < 0.5:
            cropped = torch.flip(cropped, dims=(2,))
        return cropped


def maybe_subset(dataset: Dataset, limit: int | None, seed: int) -> Dataset:
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=limit, replace=False)
    return Subset(dataset, indices.tolist())
