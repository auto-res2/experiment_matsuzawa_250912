"""src/evaluate.py
Simple visualisation helpers for COSMOS-Diff experiments.
All plots are saved under .research/iteration2/images.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # head-less rendering for CI
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
#  Helper: ensure directory exists before saving
# -----------------------------------------------------------------------------

def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
#  Plotters
# -----------------------------------------------------------------------------

def plot_training_loss(history: List[float], out_path: Path) -> None:
    """Line plot of training loss."""
    _ensure_parent(out_path)
    plt.figure(figsize=(4, 3))
    plt.plot(history, marker="o", linewidth=1.2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Critic Training Loss")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_calibration_bars(calibration: Dict[str, List[float]], out_path: Path) -> None:
    """Bar plot comparing calibration times for modes M0/M1/M2."""
    _ensure_parent(out_path)
    modes = list(calibration.keys())
    means = [sum(v) / len(v) for v in calibration.values()]
    errs = [max(v) - min(v) for v in calibration.values()]  # simple range as error bar

    plt.figure(figsize=(4, 3))
    plt.bar(modes, means, yerr=errs, capsize=4, color=["#c44", "#4c4", "#44c"])
    plt.ylabel("Calibration Wall-clock (s)")
    plt.title("Federated Cost-Model Calibration")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_uvec_violation(violation: Dict[str, float], out_path: Path) -> None:
    """Plot violation rate per ε."""
    _ensure_parent(out_path)
    eps = list(violation.keys())
    vals = list(violation.values())

    plt.figure(figsize=(4, 3))
    plt.bar(eps, vals, color="#8888ff")
    plt.ylabel("Violation Rate")
    plt.title("UVEC Bound Violations")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
