"""
preprocess.py
=============
All data-loading and preprocessing logic.  The code is intentionally
strict: if a required archive cannot be downloaded or its checksum
mismatches, execution is aborted (NO FALLBACK).
"""
from __future__ import annotations

import hashlib
import os
import sys
import tarfile
import tempfile
import zipfile
import errno
import shutil
from pathlib import Path
from typing import Tuple

import requests
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

import yaml

# ---------------------------------------------------------------------------
# Configuration helpers – the YAML files are the single source of truth.
# ---------------------------------------------------------------------------

_CFG_CACHE: dict[str, dict] = {}


def _load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if path.as_posix() not in _CFG_CACHE:
        with path.open() as fp:
            _CFG_CACHE[path.as_posix()] = yaml.safe_load(fp)
    return _CFG_CACHE[path.as_posix()]

# ---------------------------------------------------------------------------
# Verified download utilities (SHA-256 enforced, NO FALLBACK)
# ---------------------------------------------------------------------------

def _sha256(path: Path, block: int = 65536) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, sha256: str, dest_dir: Path) -> Path:
    """Download an archive with checksum verification (no silent fallbacks).

    A temporary file is first written to the default system temp directory.
    It is then *moved* into `dest_dir`.  On some CI systems `/tmp` resides
    on a different mount point than the working directory, so a plain
    `os.replace` would raise `EXDEV`.  We therefore catch this specific
    error code and fall back to a copy-and-remove sequence that is
    functionally equivalent while remaining atomic enough for our use
    case.  All other exceptions are re-raised to honour the fail-fast
    policy.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = dest_dir / os.path.basename(url.split("?", 1)[0])

    # If the file already exists and the checksum matches we are done.
    if fname.exists() and _sha256(fname) == sha256:
        return fname

    print(f"Downloading {url} …", flush=True)
    r = requests.get(url, stream=True, timeout=30)
    if r.status_code != 200:
        sys.exit(f"ERROR: {url} returned {r.status_code} – abort (NO FALLBACK)")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        for chunk in r.iter_content(1024 * 1024):  # 1-MiB chunks
            tmp.write(chunk)
    tmp_path = Path(tmp.name)

    try:
        os.replace(tmp_path, fname)
    except OSError as e:
        if e.errno == errno.EXDEV:
            # Cross-device move – fall back to copy + delete (no silent alt).
            shutil.copy2(tmp_path, fname)
            tmp_path.unlink()
        else:
            raise

    # Verify checksum post-move / copy.
    if _sha256(fname) != sha256:
        sys.exit("ERROR: SHA-256 mismatch – abort (NO FALLBACK)")
    return fname


def extract(archive: Path, dest: Path):
    if dest.exists():
        return
    print(f"Extracting {archive} …", flush=True)
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            tf.extractall(dest)
    elif zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    else:
        sys.exit("ERROR: unsupported archive format – abort")

# ---------------------------------------------------------------------------
# Dataset access – Tiny-ImageNet as minimal reproducible example.
# ---------------------------------------------------------------------------


def _wrap_dict(ds: torch.utils.data.Dataset) -> torch.utils.data.Dataset:
    """Return a *dict*-style view so that training code can stay framework-agnostic."""

    class _Dict(torch.utils.data.Dataset):
        def __init__(self, base: torch.utils.data.Dataset):
            self.base = base

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            img, label = self.base[idx]
            return {"image": img, "label": torch.tensor(label, dtype=torch.long)}

    return _Dict(ds)


def get_tiny_imagenet_dataloader(
    cfg: dict,
    train: bool = True,
) -> DataLoader:
    root = Path(cfg["common"]["data_dir"]) / "tiny_imagenet"
    url = cfg["datasets"]["tiny_imagenet"]["url"]
    sha = cfg["datasets"]["tiny_imagenet"]["sha256"]

    archive = fetch(url, sha, root.parent)
    extract(archive, root)

    split = "train" if train else "val"
    ds = torchvision.datasets.ImageFolder(root / split)

    tfms = T.Compose(
        [
            T.Resize((224, 224)),
            T.RandomCrop(224, padding=16) if train else T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ds.transform = tfms

    # Optional sub-sampling for super-fast CI smoke tests.
    subset_size = cfg["datasets"]["tiny_imagenet"].get("subset_size")
    if subset_size is not None and subset_size < len(ds):
        ds = Subset(ds, list(range(subset_size)))

    ds = _wrap_dict(ds)

    return DataLoader(
        ds,
        batch_size=cfg["common"]["batch_size"],
        shuffle=train,
        num_workers=cfg["common"].get("num_workers", 4),
        pin_memory=torch.cuda.is_available(),
    )
