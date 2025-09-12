"""src/preprocess.py
Data-loading / preprocessing helpers – not present in original script.
We only expose a no-op stub so that future work can hook real logic here.
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
    }
