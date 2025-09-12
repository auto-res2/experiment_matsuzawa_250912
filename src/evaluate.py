"""src/evaluate.py
Evaluation / analysis utilities.
As with training, the original monolithic script contained no evaluation
logic.  We therefore provide minimal stubs that keep the refactor runnable
without altering experiment behaviour.
"""
from __future__ import annotations

from typing import Dict, Any

__all__ = [
    "run_evaluation_pipeline",
]

def run_evaluation_pipeline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Dummy evaluation placeholder.

    Returns a stub dictionary with fake metrics so that the caller can persist
    JSON without errors.
    """
    return {
        "eval_status": "skipped (no evaluation code provided in reference script)",
        "accuracy": None,
    }
