# src/preprocess.py
"""Data loading and preprocessing utilities."""
from __future__ import annotations

import json
import random
import tarfile
import urllib.request
from pathlib import Path
from typing import List

import torch
import torchvision
import torchvision.transforms as T

_DATA_ROOT = Path("data")
_DATA_ROOT.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
#   Generic helpers
# -----------------------------------------------------------------------------


def _download(url: str, dest: Path) -> Path:
    """Download a file unless it already exists."""
    if dest.exists():
        return dest

    tmp = dest.with_suffix(".tmp")
    try:
        print(f"Downloading {url} → {dest} …")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
    except Exception as exc:
        raise FileNotFoundError(f"Dataset download failed for {url}: {exc}") from exc
    return dest


# -----------------------------------------------------------------------------
#   CIFAR-100 split-10 loader
# -----------------------------------------------------------------------------


class SplitCIFAR10Tasks(torch.utils.data.Dataset):
    """CIFAR-100 split into 10 sequential tasks (10 classes each)."""

    URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"

    def __init__(
        self,
        train: bool,
        task_id: int,
        transform=None,
        max_tasks: int = 10,
    ) -> None:
        super().__init__()

        archive = _download(self.URL, _DATA_ROOT / "cifar100.tar.gz")
        # torchvision will extract internally when `download=False` and archive exists.
        self._dataset = torchvision.datasets.CIFAR100(
            _DATA_ROOT, train=train, download=False
        )

        self.transform = transform or T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )

        classes_per_task = 100 // max_tasks
        low, high = task_id * classes_per_task, (task_id + 1) * classes_per_task
        self.indices = [
            i for i, (_, y) in enumerate(self._dataset) if low <= y < high
        ]

    # ------------------------------------------------------------------ dunder
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self._dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label


# -----------------------------------------------------------------------------
#   Task builder (used by main.py)
# -----------------------------------------------------------------------------

def build_tasks(ds_name: str, max_tasks: int | None = None):
    if ds_name == "cifar100_split10":
        return [SplitCIFAR10Tasks(True, t) for t in range(max_tasks or 10)]
    raise NotImplementedError(f"Dataset '{ds_name}' not yet implemented.")
