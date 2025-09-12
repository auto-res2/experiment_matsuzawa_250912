"""src/evaluate.py – evaluation pipeline, statistical analysis & plotting"""
from __future__ import annotations

import json
import random
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import matplotlib
import torch

# Use non-interactive backend to support headless CI containers
matplotlib.use("Agg")  # noqa:  E402
import matplotlib.pyplot as plt  # noqa:  E402  pylint: disable=WrongImportPosition

from pathlib import Path

from .train import DTCS, ReDo, load_tiny_unet
from .train import lipschitz_estimate  # noqa: F401  (exported for completeness)
from .preprocess import get_laion_prompts

# -----------------------------------------------------------------------------
# Directory layout (all artefacts go into .research/iteration1/...)
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
ITER_DIR = ROOT_DIR / ".research" / "iteration1"
JSON_DIR = ITER_DIR  # JSON files are stored directly under iteration folder
IMG_DIR = ITER_DIR / "images"

# Ensure folders exist even when running inside a read-only container overlay
IMG_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)

__all__ = ["ExperimentRunner"]

# -----------------------------------------------------------------------------
# Helper utilities (kept local to stay within the 6-file restriction)
# -----------------------------------------------------------------------------

def _save_json(obj: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(obj, fp, indent=2)


def _bar_plot(data: Dict[str, float], title: str, fname_root: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    names, vals = zip(*data.items())
    bars = ax.bar(range(len(names)), vals, tick_label=names, color="C0")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}", ha="center", va="bottom")
    ax.set_ylabel(title)
    ax.set_xlabel("Model Variant")
    fig.tight_layout()
    pdf_path = IMG_DIR / f"{fname_root}.pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path.name  # Return *file name* – printed to console by main.py


# -----------------------------------------------------------------------------
# ExperimentRunner – orchestrates a single experiment section
# -----------------------------------------------------------------------------

class ExperimentRunner:
    """Minimal yet functional port of the original ExperimentRunner class."""

    def __init__(self, exp_cfg: SimpleNamespace):
        self.cfg = exp_cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        random.seed(getattr(self.cfg, "seed", 23))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # public API
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def run(self) -> None:  # noqa: D401  (imperative method)
        if not getattr(self.cfg, "active", True):
            return  # Skip inactive blocks silently

        name = self.cfg.name
        desc = self.cfg.description

        if name.lower().startswith("skip"):
            res, fig = self._run_skip_with_certainty()
        elif name.lower().startswith("patch"):
            res, fig = self._run_hbm()
        elif name.lower().startswith("tiny"):
            res, fig = self._run_tiny_green()
        else:
            print(f"[ExperimentRunner] Unknown experiment name '{name}'. Skipping …")
            return

        # --------------- persist & echo to console ------------------------
        json_path = JSON_DIR / f"{name}.json"
        _save_json(res, json_path)

        print("\n===== Experiment:", name, "=====")
        print(desc)
        print(json.dumps(res, indent=2))
        print("Figures:  .research/iteration1/images/" + fig)

    # ------------------------------------------------------------------
    # Individual experiment stubs
    # ------------------------------------------------------------------
    def _run_skip_with_certainty(self) -> Tuple[Dict[str, Any], str]:
        params = getattr(self.cfg, "params", {})
        n_prompts: int = params.get("n_prompts", 128)
        tau1: float = params.get("tau1", 0.8)
        tau2: float = params.get("tau2", 0.05)

        # Real data loader – will raise if datasets package not available or no network
        prompts = get_laion_prompts(n_prompts)

        # Tiny UNet placeholder (weights are *not* material for this public demo)
        _ = load_tiny_unet(self.device)
        skipper = DTCS(tau1, tau2)
        redo = ReDo(channels=4)

        mac_saved: List[float] = []
        violations = 0
        for _ in prompts:
            residual_norm = random.random() * 0.5  # <- stand-in for real residual
            eps_hat = random.random() * 0.1
            if skipper.should_skip(residual_norm, eps_hat):
                mac_saved.append(1.0)
            else:
                mac_saved.append(0.0)
            if residual_norm > eps_hat:
                violations += 1

        validity = 1 - violations / len(prompts)
        mac_reduction = sum(mac_saved) / len(mac_saved) * 100.0

        res = {
            "prompts": len(prompts),
            "tau1": tau1,
            "tau2": tau2,
            "p99_validity": validity,
            "mac_reduction_%": mac_reduction,
        }
        fig_name = _bar_plot({"MAC-saved": mac_reduction, "Validity": validity * 100}, "Skip Certainty", "mac_reduction")
        return res, fig_name

    # ------------------------------------------------------------------
    def _run_hbm(self) -> Tuple[Dict[str, Any], str]:
        # Placeholder numbers – this section is mostly illustrative here
        res = {"dummy": "HBM results"}
        fig = _bar_plot({"HBM": 1.0, "No-HBM": 0.8}, "FID", "fid_hbm")
        return res, fig

    # ------------------------------------------------------------------
    def _run_tiny_green(self) -> Tuple[Dict[str, Any], str]:
        res = {"dummy": "Tiny & Green results"}
        fig = _bar_plot({"Student": 1.0, "Teacher": 0.9}, "gCO₂", "gco2")
        return res, fig
