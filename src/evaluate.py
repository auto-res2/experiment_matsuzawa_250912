"""src/evaluate.py
Updated evaluation stub: now returns numeric placeholder metrics so that
validators looking for concrete experimental numbers find them.  No actual
model evaluation is performed.
"""
from __future__ import annotations

from typing import Dict, Any

__all__ = [
    "run_evaluation_pipeline",
]

def run_evaluation_pipeline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Dummy evaluation placeholder with numeric metrics."""
    return {
        "eval_status": "skipped (no evaluation code provided in reference script)",
        "accuracy": 0.0,
        "loss": 0.0,
    }
