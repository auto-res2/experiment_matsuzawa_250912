"""src/preprocess.py – data acquisition & preprocessing utilities"""
from __future__ import annotations

import itertools
import random
from typing import List

from pathlib import Path

# datasets is lazy-imported so that preprocessing utilities work even when the
# optional dependency is missing for certain CI configurations.

def _require_datasets() -> None:  # Raises informative RuntimeError if unavailable
    try:
        import datasets  # noqa:  F401  (import for side-effects)
    except ModuleNotFoundError as exc:  # pragma:   no-cover
        raise RuntimeError(
            "The 'datasets' package is required for data loading. "
            "You can install optional dependencies via:  pip install datasets"
        ) from exc


# -----------------------------------------------------------------------------
# Global data directory (unused at the moment but kept for completeness)
# -----------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# Public loader functions
# -----------------------------------------------------------------------------

def get_laion_prompts(n: int, split: str = "train") -> List[str]:
    """Stream *n* text prompts from the LAION aesthetics subset."""
    _require_datasets()
    from datasets import load_dataset  # pylint: disable=import-error

    try:
        ds = load_dataset("laion/laion2B-en-aesthetic", streaming=True, split=split)
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"Cannot access LAION dataset – {exc}") from exc

    prompts: List[str] = []
    for sample in itertools.islice(ds, n):
        prompts.append(sample["TEXT"])

    if len(prompts) < n:
        raise RuntimeError(f"Requested {n} prompts, only {len(prompts)} retrieved.")
    return prompts


def get_coco_images(n: int, split: str = "train") -> List[str]:
    """Return *n* image file paths from MS-COCO captions."""
    _require_datasets()
    from datasets import load_dataset  # pylint: disable=import-error

    try:
        ds = load_dataset("coco_captions", split=split)
    except Exception as exc:
        raise RuntimeError(f"Cannot access MS-COCO – {exc}") from exc

    # The streaming option is not yet supported for COCO, hence we shuffle & select.
    paths: List[str] = []
    for rec in ds.shuffle(seed=17).select(range(n)):
        # The HF iterator returns PIL images – we extract the underlying filename
        paths.append(rec["image"].filename)
    return paths
