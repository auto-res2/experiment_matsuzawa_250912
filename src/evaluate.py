# src/evaluate.py
"""Simple evaluation helpers used by main.py."""
from __future__ import annotations

from typing import Dict


def log_metrics(split_name: str, metrics: Dict[str, float]):
    """Pretty-print metrics to stdout."""
    joined = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    print(f"[{split_name}] {joined}")
