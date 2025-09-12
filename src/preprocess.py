"""
preprocess.py – Data downloading and verification utilities.  These functions
are purposely stringent: any uncertainty (missing URL, size or checksum) leads
to an immediate, explicit RuntimeError in compliance with the paper‘s
STRICT NO-FALLBACK RULE.
"""
from __future__ import annotations

import hashlib
import shutil
import tarfile
from pathlib import Path
from typing import Dict, Any

import requests
from tqdm import tqdm

__all__ = [
    "DataUnavailableError",
    "fetch_dataset",
]


class DataUnavailableError(RuntimeError):
    """Raised when a dataset cannot be downloaded or verified."""


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

_CHUNK = 1 << 20  # 1 MiB


def _download(url: str, target: Path, expected_size: int, sha256: str):
    """Download *url* → *target* and verify size / SHA-256 checksum."""

    with requests.get(url, stream=True, timeout=30) as r:
        if r.status_code != 200:
            raise DataUnavailableError(f"Cannot download {url} – HTTP {r.status_code}")
        total = int(r.headers.get("content-length", 0))
        if expected_size and total and total != expected_size:
            raise DataUnavailableError("Remote file size mismatch – aborting.")

        with target.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for chunk in r.iter_content(chunk_size=_CHUNK):
                f.write(chunk)
                pbar.update(len(chunk))

    # ---------------------------------------------------------------------
    # Checksum verification
    if sha256:
        m = hashlib.sha256()
        with target.open("rb") as f:
            for chunk in iter(lambda: f.read(_CHUNK), b""):
                m.update(chunk)
        if m.hexdigest() != sha256.lower():
            raise DataUnavailableError("SHA-256 mismatch – dataset corrupted.")


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def fetch_dataset(name: str, spec: Dict[str, Any], data_root: Path) -> Path:
    """Download/extract *name* according to *spec* into *data_root*.

    Returns the path to the extracted dataset.  Raises DataUnavailableError on
    any failure.
    """

    url: str | None = spec.get("url")
    if not url:
        raise DataUnavailableError(
            f"Dataset '{name}' has no download URL – cannot continue."
        )

    file_name = Path(url).name
    dl_path = data_root / file_name
    if not dl_path.exists():
        _download(url, dl_path, spec.get("size_bytes", 0), spec.get("sha256", ""))

    # Auto-extract archives
    extract_dir = data_root / name
    if tarfile.is_tarfile(dl_path):
        with tarfile.open(dl_path) as tar:
            tar.extractall(path=extract_dir)
    elif dl_path.suffix == ".zip":
        shutil.unpack_archive(str(dl_path), extract_dir=str(extract_dir))

    return extract_dir
