"""
preprocess.py
~~~~~~~~~~~~~
Pure data-handling, privacy utilities and misc. helper functions that are shared
across *train* / *evaluate* / *main*.
"""
from __future__ import annotations

import math
import random
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Dict

import json
import yaml
import numpy as np
import requests
import torch
from opacus.accountants.rdp import RDPAccountant

# ---------------------------------------------------------------------------
#  Dataset management
# ---------------------------------------------------------------------------
DATASETS: Dict[str, Dict[str, str]] = {
    "reddit_threads": {
        "url": "https://snap.stanford.edu/graphsage/reddit.zip",
        "compressed": "reddit.zip",
        "processed_folder": "reddit_threads",
    },
    "elliptic_btc": {
        "url": "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set",
        "compressed": "elliptic.zip",
        "processed_folder": "elliptic_btc",
    },
    "taobao_actions": {
        "url": "https://tianchi.aliyun.com/download/ae4812a34b784ef889d361b5d54cf29f",
        "compressed": "taobao.tgz",
        "processed_folder": "taobao_actions",
    },
    # "ring_transfer_stream" is generated on-the-fly in the experiment code.
}


class DataManager:
    """Downloads and prepares datasets (STRICT NO-FALLBACK)."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _download(self, url: str, tgt: Path):
        print(f"Downloading {url} → {tgt} …")
        with requests.get(url, stream=True) as r:
            if r.status_code != 200:
                raise RuntimeError(
                    f"Download failed with status {r.status_code}. "
                    "Check authentication or dataset availability."
                )
            with open(tgt, "wb") as fh:
                for chunk in r.iter_content(chunk_size=8192):
                    fh.write(chunk)

    # ------------------------------------------------------------------
    def _extract(self, archive: Path, dst: Path):
        print(f"Extracting {archive.name} …")
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(dst)
        elif archive.suffix in {".tgz", ".gz"}:
            with tarfile.open(archive) as tf:
                tf.extractall(dst)
        else:
            raise RuntimeError(f"Unsupported archive format: {archive.suffix}")

    # ------------------------------------------------------------------
    def fetch(self, name: str) -> Path:
        if name not in DATASETS:
            raise ValueError(f"Unknown dataset '{name}'")
        meta = DATASETS[name]
        processed_dir = self.root / meta["processed_folder"]
        if processed_dir.exists() and any(processed_dir.iterdir()):
            return processed_dir  # already present

        compressed = self.root / meta["compressed"]
        self._download(meta["url"], compressed)
        self._extract(compressed, processed_dir)
        return processed_dir


# ---------------------------------------------------------------------------
#  Determinism utilities
# ---------------------------------------------------------------------------

def set_seeds(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------------------
#  I/O helpers
# ---------------------------------------------------------------------------

def save_json(obj: Any, path: Path):
    path.write_text(json.dumps(obj, indent=2))


def save_yaml(obj: Any, path: Path):
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(obj, fh)


# ---------------------------------------------------------------------------
#  Privacy helpers (JL + Gaussian mechanism + RDP accountant)
# ---------------------------------------------------------------------------
class JLProjector(torch.nn.Module):
    """Gaussian Johnson–Lindenstrauss projection."""

    def __init__(self, in_dim: int, out_dim: int, eps: float = 0.1):  # noqa: D401
        super().__init__()
        self.register_buffer("W", torch.randn(out_dim, in_dim) / math.sqrt(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.t()


def add_dp_noise(x: torch.Tensor, epsilon: float, delta: float, sensitivity: float = 1.0) -> torch.Tensor:
    sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    noise = torch.randn_like(x) * sigma
    return x + noise


class PrivacyAccountant:
    """Very small wrapper around Opacus' RDP accountant."""

    def __init__(self, epsilon_target: float, delta: float):
        self.acc = RDPAccountant()
        self.epsilon_target = epsilon_target
        self.delta = delta

    def step(self, noise_multiplier: float, sample_rate: float, steps: int):
        self.acc.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate, steps=steps)

    def privacy_spent(self) -> float:
        eps, _ = self.acc.get_privacy_spent(self.delta)
        return eps
