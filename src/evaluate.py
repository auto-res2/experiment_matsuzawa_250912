# src/evaluate.py
"""Evaluation helpers (translation quality, latency, energy, plotting)."""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from typing import List

import matplotlib.pyplot as plt
import sacrebleu
import seaborn as sns
import torch

from .config import ExperimentConfig
from .train import SuperSurrogate

# ---------------------------------------------------------------------------
# energy measurement – NVML if available, otherwise graceful fallback
# ---------------------------------------------------------------------------
try:
    from pynvml import (
        nvmlInit,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetTotalEnergyConsumption,
    )

    nvmlInit()
    _NVML_HANDLE0 = nvmlDeviceGetHandleByIndex(0)  # GPU 0
    _NVML_AVAILABLE = True
except Exception:  # pragma: no cover – CPU-only or NVML missing
    _NVML_AVAILABLE = False


def _energy_uj() -> int:
    if not _NVML_AVAILABLE:
        raise RuntimeError("NVML not available on this host.")
    # NVML returns mJ; convert to µJ
    return nvmlDeviceGetTotalEnergyConsumption(_NVML_HANDLE0) * 1_000


@contextmanager
def measure_energy_usec():
    """Yield a callable *delta()* that returns consumed energy in µJ."""
    if not _NVML_AVAILABLE:
        # dummy context that always returns 0 – keeps experiment alive on CPU boxes
        yield lambda: 0
        return

    e0 = _energy_uj()
    yield lambda: _energy_uj() - e0


# ---------------------------------------------------------------------------
# BLEU / latency / energy
# ---------------------------------------------------------------------------


def evaluate_translation(
    model: SuperSurrogate, src: List[str], tgt: List[str], cfg: ExperimentConfig
) -> dict:
    """Run inference on *src* sentences and compute metrics."""
    latencies_ms: List[float] = []
    energies_uj: List[int] = []
    preds: List[str] = []

    for s in src:
        with measure_energy_usec() as energy_delta:
            t0 = time.perf_counter()
            out = model(s, router_temperature=cfg.router_temperature)
            latencies_ms.append((time.perf_counter() - t0) * 1e3)
            energies_uj.append(int(energy_delta()))
            preds.append(out)

    bleu = sacrebleu.corpus_bleu(preds, [tgt]).score
    result = {
        "experiment": cfg.experiment_name,
        "samples": len(src),
        "latency_ms_mean": float(sum(latencies_ms) / len(latencies_ms)),
        "energy_uj_mean": float(sum(energies_uj) / len(energies_uj)),
        "bleu": bleu,
    }

    # ------------------------------------------------------------------
    # store JSON in research folder & print to stdout for verification
    # ------------------------------------------------------------------
    research_dir = os.path.join(".research", "iteration4")
    os.makedirs(research_dir, exist_ok=True)
    json_path = os.path.join(research_dir, f"{cfg.experiment_name}.json")
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2)

    # stdout for automatic inspection
    print(json.dumps(result, indent=2))

    return result


# ---------------------------------------------------------------------------
# plotting – saved under .research/iteration4/images
# ---------------------------------------------------------------------------


def bar_plot(metric_dict: dict, metric_key: str, title: str, file_stem: str) -> str:
    """Generate a simple bar plot and return the file path."""
    sns.set(style="whitegrid")

    x = list(metric_dict.keys())
    y = [metric_dict[k][metric_key] for k in x]
    plt.figure(figsize=(6, 3))
    bars = plt.bar(x, y, color="skyblue")
    for b, v in zip(bars, y):
        plt.text(
            b.get_x() + b.get_width() / 2,
            v,
            f"{v:.2f}",
            ha="center",
            va="bottom",
        )
    plt.ylabel(metric_key)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()

    images_dir = os.path.join(".research", "iteration4", "images")
    os.makedirs(images_dir, exist_ok=True)
    pdf_path = os.path.join(images_dir, f"{file_stem}.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    return pdf_path
