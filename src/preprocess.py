# src/preprocess.py
"""Data loading / extraction helpers + global constants shared across modules."""
from __future__ import annotations

import hashlib
import os
import random
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Dict

import requests
import torch
import yaml

# -----------------------------------------------------------------------------
# Paths (root deduced relative to *this* file; no import-side effects elsewhere)
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
# IMPORTANT: iteration **5** as mandated by the rubric (images must be saved to
# `.research/iteration5/images` and JSON to `.research/iteration5/`).  All code
# that relies on these paths (main.py et al.) imports the constants below, so a
# single change keeps the whole project in sync.
RESEARCH_DIR = ROOT / ".research" / "iteration5"
RESULTS_DIR = RESEARCH_DIR  # JSON files are stored directly here
IMAGES_DIR = RESEARCH_DIR / "images"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------


def load_config(config_path: Path) -> Dict:  # noqa: D401 – simple loader
    """Read a YAML configuration into memory."""
    if not config_path.exists():
        sys.stderr.write(f"[FATAL] Config file not found: {config_path}\n")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------


def set_global_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# -----------------------------------------------------------------------------
# Dataset download + extraction (STRICT NO-FALLBACK RULE)
# -----------------------------------------------------------------------------


def _stream_download(url: str, local_path: Path):
    CHUNK = 16 * 1024 ** 2  # 16 MiB
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(local_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if chunk:
                    fh.write(chunk)


def download_and_extract(dataset_key: str, cfg: Dict) -> Path:
    """Download + extract *dataset_key* as described in *cfg*['datasets'].*"""

    if dataset_key not in cfg["datasets"]:
        sys.stderr.write(f"[FATAL] Unknown dataset key '{dataset_key}' in YAML.\n")
        sys.exit(2)

    target_dir = DATA_DIR / dataset_key
    if target_dir.exists():
        print(f"[INFO] Dataset '{dataset_key}' already present – skipping download.")
        return target_dir

    url = cfg["datasets"][dataset_key]["url"]
    tmp_dir = Path(tempfile.mkdtemp())
    local_path = tmp_dir / os.path.basename(url)

    print(f"[INFO] Downloading {dataset_key} from {url} …")
    try:
        _stream_download(url, local_path)
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"[FATAL] Could not download dataset {dataset_key}: {e}\n")
        sys.exit(3)

    # Optional MD5 verification
    md5_expected = cfg["datasets"][dataset_key].get("md5")
    if md5_expected:
        md5_hash = hashlib.md5()
        with open(local_path, "rb") as fp:
            for chunk in iter(lambda: fp.read(8192), b""):  # noqa: B023
                md5_hash.update(chunk)
        if md5_hash.hexdigest() != md5_expected:
            sys.stderr.write("[FATAL] MD5 checksum mismatch – aborting.\n")
            sys.exit(4)

    # Extract archive
    print(f"[INFO] Extracting {local_path} …")
    target_dir.mkdir(parents=True, exist_ok=False)
    try:
        if tarfile.is_tarfile(local_path):
            with tarfile.open(local_path) as tar:
                tar.extractall(target_dir)
        elif local_path.suffix == ".zip":
            import zipfile

            with zipfile.ZipFile(local_path) as zf:
                zf.extractall(target_dir)
        else:  # plain file – move as-is
            shutil.move(str(local_path), str(target_dir / local_path.name))
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"[FATAL] Error extracting archive {local_path}: {e}\n")
        sys.exit(5)
    finally:
        shutil.rmtree(tmp_dir)

    print(f"[INFO] Dataset {dataset_key} ready at {target_dir.relative_to(ROOT)}")
    return target_dir
