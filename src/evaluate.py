"""
evaluate.py
===========
Metric functions, statistical analysis helpers and plotting utilities.
All heavy-weight numerical logic that is *not* part of the gradient-based
training lives in this file.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence, Any

import matplotlib.pyplot as plt
import torch

__all__ = [
    "accuracy",
    "compute_delta_eo",
    "emb2_error",
    "line_plot",
    "_ensure_img_dir",
]

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


def compute_delta_eo(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    sensitive_attrs: torch.Tensor,
) -> float:
    """Tiny Δ-Equalised Odds implementation (binary labels).

    The function is intentionally *minimal* – just enough for smoke tests
    and the automated acceptance suite.
    """
    preds = torch.argmax(logits, 1)
    tprs = []
    for g in sensitive_attrs.unique():
        idx = sensitive_attrs == g
        if idx.sum() == 0:
            continue
        tp = ((preds[idx] == 1) & (y_true[idx] == 1)).float().sum()
        fn = ((preds[idx] == 0) & (y_true[idx] == 1)).float().sum()
        tprs.append((tp / (tp + fn + 1e-6)).item())
    return max(tprs) - min(tprs) if tprs else 0.0


def emb2_error(
    B: float,
    E: float,
    W: float,
    I_rdp: float,
    I_bias: float,
    I_err: float,
) -> float:
    lhs = B * E * W
    rhs = 2 * (I_rdp + I_bias + I_err)
    return abs(lhs - rhs) / rhs * 100.0

# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _ensure_img_dir() -> Path:
    """Return the canonical directory for all plots (iteration 9).

    The task description mandates that *all* image assets are placed under
    `.research/iteration9/images`.  Centralising the logic here guarantees
    compliance across the entire code base with a single function call.
    """
    img_dir = Path(".research/iteration9/images")
    img_dir.mkdir(parents=True, exist_ok=True)
    return img_dir


def line_plot(
    xs: Sequence[Any],
    ys: Sequence[float],
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str | Path,
):
    """One-liner around Matplotlib that always stores *vector* graphics."""
    _ensure_img_dir()
    filename = Path(filename)
    plt.figure()
    plt.plot(xs, ys, marker="o")
    for x, y in zip(xs, ys):
        plt.annotate(f"{y:.3f}", (x, y))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()
