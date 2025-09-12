"""
preprocess.py – dataset acquisition / preparation utilities (verbatim from the
original src/data.py).
"""
from __future__ import annotations

import hashlib
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Any

from tqdm import tqdm

__all__ = ["ensure_datasets_present"]

# ---------------------------------------------------------------------------
#  Internal helpers – unchanged
# ---------------------------------------------------------------------------
class _ProgressBar(tqdm):
    def update_to(self, b: int = 1, bsize: int = 1, tsize: int | None = None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def _download(url: str, dest: Path, sha256: str | None = None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    with _ProgressBar(unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]) as t:
        urllib.request.urlretrieve(url, filename=dest, reporthook=t.update_to)
    if sha256 is not None:
        h = hashlib.sha256(dest.read_bytes()).hexdigest()
        if h != sha256:
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"SHA-256 mismatch for {dest}")


def _extract(archive: Path, out_dir: Path):
    if out_dir.exists():
        return
    print(f"[data] Extracting {archive.name} …")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(out_dir)
    elif archive.suffix in {".tgz", ".gz"} or archive.suffixes[-2:] == [".tar", ".gz"]:
        with tarfile.open(archive) as t:
            t.extractall(out_dir)
    else:
        raise RuntimeError(f"Unknown archive format {archive}")

# ---------------------------------------------------------------------------
#  Public API – ensure that all datasets referenced in CONFIG exist locally
# ---------------------------------------------------------------------------

def ensure_datasets_present(cfg: Dict[str, Any], data_root: Path):
    # -------------------- MS-COCO 2017 -----------------------------------
    coco = cfg["datasets"]["coco2017"]
    coco_img_zip = data_root / "coco_val2017.zip"
    coco_ann_zip = data_root / "coco_ann2017.zip"
    _download(coco["urls"]["images"], coco_img_zip)
    _download(coco["urls"]["captions"], coco_ann_zip)
    _extract(coco_img_zip, data_root / "coco2017" / "images")
    _extract(coco_ann_zip, data_root / "coco2017" / "annotations")

    # -------------------- WMT22 EN-ZH ------------------------------------
    wmt = cfg["datasets"]["wmt22_en_zh"]
    wmt_tar = data_root / "wmt22.tgz"
    _download(wmt["url"], wmt_tar)
    _extract(wmt_tar, data_root / "wmt22")

    # -------------------- Habitat-Lite -----------------------------------
    hab_dir = data_root / "habitat_lite"
    if not hab_dir.exists():
        print("[data] Cloning Habitat-Sim (lite)…")
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                cfg["datasets"]["habitat_lite"]["git"],
                str(hab_dir),
            ],
            check=True,
        )
