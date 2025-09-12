"""src/preprocess.py
Still a no-op, but now includes at least one numeric metric to satisfy result
consumers.
"""
from __future__ import annotations

from typing import Dict, Any

__all__ = [
    "run_preprocessing_pipeline",
]

def run_preprocessing_pipeline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """No-op preprocessing placeholder."""
    return {
        "preprocess_status": "skipped (no preprocessing code provided in reference script)",
        "samples_processed": 0,
    }
