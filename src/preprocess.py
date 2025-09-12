"""src/preprocess.py – data acquisition & dataset helpers."""
from __future__ import annotations

import pathlib
import tarfile
import zipfile
from typing import Dict

import requests
import tqdm
from torchvision import datasets, transforms

CHUNK = 8192

# ----------------------------------------------------------------------------
#  Download & extraction helpers
# ----------------------------------------------------------------------------

def _download(url: str, out_path: pathlib.Path) -> None:
    if out_path.exists():
        return  # already downloaded
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[DOWNLOAD] {url} → {out_path}")
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with tqdm.tqdm(total=total, unit="B", unit_scale=True) as pbar:
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(CHUNK):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

def _extract(archive: pathlib.Path, dst: pathlib.Path) -> None:
    print(f"[EXTRACT] {archive} → {dst}")
    dst.mkdir(parents=True, exist_ok=True)
    if archive.suffix in {".tgz", ".gz", ".tar"}:
        with tarfile.open(archive, "r:*") as tar:
            tar.extractall(dst)
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as z:
            z.extractall(dst)
    else:  # pragma: no cover
        raise RuntimeError(f"Unknown archive type for {archive}")

def assure_datasets(ds_cfg: Dict[str, Dict[str, str]], data_root: pathlib.Path) -> Dict[str, pathlib.Path]:
    """Ensure all datasets are present on disk; returns mapping name → local path."""

    out: Dict[str, pathlib.Path] = {}
    for name, meta in ds_cfg.items():
        url = meta["url"]
        file_name = url.split("/")[-1]
        archive = data_root / "_raw" / file_name
        _download(url, archive)
        dst = data_root / name
        if not dst.exists() or not any(dst.iterdir()):
            _extract(archive, dst)
        out[name] = dst
    return out

# ----------------------------------------------------------------------------
#  Simple dataset split helpers (CIFAR-100 split-10)
# ----------------------------------------------------------------------------

def cifar100_split10(root: pathlib.Path, task_id: int, train: bool):  # noqa: D401
    """Returns a dataset containing 10 CIFAR-100 classes per task."""

    transform_list = [
        transforms.RandomCrop(32, padding=4) if train else transforms.CenterCrop(32),
        transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]

    ds = datasets.CIFAR100(root=root, download=False, train=train, transform=transforms.Compose(transform_list))
    cls_range = list(range(task_id * 10, (task_id + 1) * 10))
    idx = [i for i, y in enumerate(ds.targets) if y in cls_range]
    ds.targets = [ds.targets[i] for i in idx]
    ds.data = ds.data[idx]
    return ds
