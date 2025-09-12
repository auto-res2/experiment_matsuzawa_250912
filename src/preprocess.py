from __future__ import annotations

"""src/preprocess.py
Data-loading utilities.  They intentionally fail fast if a dataset is not
accessible – the original code followed a strict no-fallback policy which we
retain.
"""

import sys
from typing import Any

from datasets import load_dataset

__all__ = [
    "get_wmt14",
    "sanity_check_dataset_access",
]


def get_wmt14(split: str, proportion: float = 1.0):
    """Return a (potentially down-sampled) slice of WMT14 En↔De.

    The function guarantees that at least *one* element is returned whenever
    the underlying dataset is accessible so that downstream code never runs
    into empty-slice statistics during a smoke test.
    """
    try:
        ds = load_dataset("wmt14", "de-en", split=split)
        if proportion < 1.0:
            # Guarantee ≥1 example so that the smoke test cannot under-sample to 0.
            k = max(1, int(len(ds) * proportion))
            ds = ds.shuffle(seed=42).select(range(k))
        return ds
    except Exception as exc:  # pragma: no cover – strict fail-fast policy
        sys.exit(f"[ERROR] Failed to prepare WMT14: {exc}")


def sanity_check_dataset_access(name: str):
    """Quick 1 % load to verify that credentials are present for gated sets."""
    try:
        _ = load_dataset(name, split="train[:1%]")
    except Exception:
        sys.exit(
            f"[ERROR] Dataset '{name}' is not accessible in the current environment. "
            "Per the no-fallback rule, terminating."
        )
