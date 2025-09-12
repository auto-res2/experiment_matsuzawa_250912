# src/preprocess.py
"""Data loading and preprocessing utilities."""
from __future__ import annotations

import random
from pathlib import Path
from typing import List

import torch
import torchvision
import torchvision.transforms as T

_DATA_ROOT = Path("data")
_DATA_ROOT.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
#   Stub download helper (strict – fail-fast)
# -----------------------------------------------------------------------------


def _download(url: str, dest: Path) -> Path:
    """Download a file unless it already exists."""
    if dest.exists():
        return dest

    raise FileNotFoundError(
        f"Dataset download blocked in this environment. Expected file at {dest}."
    )


# -----------------------------------------------------------------------------
#   CIFAR-100 split-10 loader (used for full experiment)
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

        # Fail-fast download (no silent fallbacks)
        archive = _download(self.URL, _DATA_ROOT / "cifar100.tar.gz")
        self._dataset = torchvision.datasets.CIFAR100(_DATA_ROOT, train=train, download=False)

        self.transform = transform or T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )

        classes_per_task = 100 // max_tasks
        low, high = task_id * classes_per_task, (task_id + 1) * classes_per_task
        self.indices = [i for i, (_, y) in enumerate(self._dataset) if low <= y < high]

    # ------------------------------------------------------------------ dunder
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self._dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label


# -----------------------------------------------------------------------------
#   FAKE tiny dataset for smoke-tests (no external download)
# -----------------------------------------------------------------------------


class _FakeTaskDataset(torch.utils.data.Dataset):
    """Very small synthetic dataset (100 samples, 3×32×32) per task."""

    def __init__(self, task_id: int, samples: int = 100, num_tasks: int = 10):
        super().__init__()
        self.task_id = task_id
        self.samples = samples
        self.num_tasks = num_tasks
        self.rng = random.Random(42 + task_id)  # deterministic per task

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        # Images in [0,1]
        img = torch.rand(3, 32, 32)
        label = self.rng.randint(0, 9) + self.task_id * 10  # unique label space per task
        return img, label


# -----------------------------------------------------------------------------
#   Task builder (used by main.py)
# -----------------------------------------------------------------------------


def build_tasks(ds_name: str, max_tasks: int | None = None):
    if ds_name == "cifar100_split10":
        return [SplitCIFAR10Tasks(True, t) for t in range(max_tasks or 10)]
    if ds_name == "fake_small":
        t = max_tasks or 2
        return [_FakeTaskDataset(task_id=i) for i in range(t)]
    raise NotImplementedError(f"Dataset '{ds_name}' not yet implemented.")
