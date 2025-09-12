"""
preprocess.py – dataset download & extraction utilities
───────────────────────────────────────────────────────
Contains the *exact* functionality from the monolithic script, only relocated
into its own module so that it can be imported by both training and evaluation
code without circular dependencies.
"""
from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Dict

from tqdm import tqdm
import logging

logger = logging.getLogger("ReFuse-CL")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

__all__ = [
    "fetch_dataset",
]

# -----------------------------------------------------------------------------
#  Internal helper: progress-bar enabled file download
# -----------------------------------------------------------------------------

def _download_with_progress(url: str, dest: Path) -> None:
    class _TqdmUpTo(tqdm):
        def update_to(self, b: int = 1, bsize: int = 1, tsize: int | None = None):  # noqa: N802
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    try:
        with _TqdmUpTo(
            unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=dest.name
        ) as t:
            urllib.request.urlretrieve(url, filename=str(dest), reporthook=t.update_to)
    except Exception as err:  # noqa: BLE001 – broad except OK in fail-fast util
        raise RuntimeError(
            f"Failed to download required dataset from {url}. "
            f"Terminating as per STRICT NO-FALLBACK rule.  Original error: {err}"
        ) from err


# -----------------------------------------------------------------------------
#  Public API: fetch & extract dataset
# -----------------------------------------------------------------------------

def fetch_dataset(cfg: Dict[str, Any]) -> Path:
    """Download (if necessary) and extract the DR-CL-Bench archive.

    Returns
    -------
    Path
        Root directory containing the extracted dataset.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = DATA_DIR / Path(cfg["dataset_url"]).name

    if not archive_path.exists():
        logger.info("Downloading dataset archive from  %s  …", cfg["dataset_url"])
        _download_with_progress(cfg["dataset_url"], archive_path)
    else:
        logger.info("Dataset archive already present – skipping download.")

    extract_root = DATA_DIR / "dr_cl_bench_v3"
    if extract_root.exists() and any(extract_root.iterdir()):
        logger.info("Dataset already extracted – skipping extraction.")
        return extract_root

    logger.info("Extracting dataset … (this may take a while)")
    try:
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(extract_root)
    except Exception as err:  # noqa: BLE001 – broad except OK here
        raise RuntimeError(
            f"Failed to extract dataset archive {archive_path}. "
            f"Aborting as per STRICT NO-FALLBACK rule. Error: {err}"
        ) from err

    logger.info("Dataset extracted to %s", extract_root)
    return extract_root
