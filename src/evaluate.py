"""src/evaluate.py
Evaluation utilities, quality metrics and the primary Experiment-1 benchmark
(text track on WMT14).  All heavy computation happens here so that *main.py*
remains a thin orchestrator.
"""

from __future__ import annotations

import json
import time
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sacrebleu import corpus_bleu
from transformers import AutoTokenizer

# Third-party helpers that live in this repository
from .train import load_hf_model
from .preprocess import get_wmt14

# -----------------------------------------------------------------------------
# I/O locations – fixed relative to project root
# -----------------------------------------------------------------------------
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT_DIR / ".research" / "iteration1"
IMAGES_DIR = RESULT_DIR / "images"
for _d in (RESULT_DIR, IMAGES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Optional power/thermal support via NVML
# -----------------------------------------------------------------------------
try:
    import pynvml  # noqa: WPS433  (external dependency)

    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:  # pragma: no cover – NVML may simply be absent
    _NVML_AVAILABLE = False


def _energy_reading_j() -> float:
    """Return total energy in joules (0.0 if unsupported)."""
    if not _NVML_AVAILABLE:
        return 0.0
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    # NVML delivers milli-joules; convert to J for readability
    return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) / 1000.0


# -----------------------------------------------------------------------------
# Generic metric helpers
# -----------------------------------------------------------------------------


def compute_bleu(preds: List[str], refs: List[str]) -> float:  # noqa: D401
    """Lightweight BLEU wrapper so we do not import the full sacrebleu API."""
    return corpus_bleu(preds, [refs]).score


@dataclass
class ExampleStats:
    latency_ms: float
    energy_j: float
    quality: float  # BLEU score for this paper – could hold anything


# -----------------------------------------------------------------------------
# PRIMARY BENCHMARK – EXPERIMENT 1
# -----------------------------------------------------------------------------


def experiment1(cfg: Dict[str, Any]) -> Dict[str, float]:
    """Latency / Energy / Quality on WMT14 En→De (teacher vs. QuADRoN-DM)."""

    exp_name = "experiment1_text_wmt14"
    tokenizer = AutoTokenizer.from_pretrained(cfg["teacher_model"])
    teacher = load_hf_model(cfg["teacher_model"])  # not used yet but ready for extensions
    quadron = load_hf_model(cfg["quadron_model"])

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    ds = get_wmt14(cfg["dataset_split"], proportion=cfg["dataset_proportion"])
    stats: List[ExampleStats] = []

    device = next(quadron.parameters()).device  # robust way to detect placement

    # ------------------------------------------------------------------
    # Main loop – sample-level measurements for fine-grained histograms.
    # ------------------------------------------------------------------
    for ex in ds:
        src_txt = ex["translation"]["en"]
        ref_txt = ex["translation"]["de"]

        inputs = tokenizer(src_txt, return_tensors="pt").to(device)
        e_before = _energy_reading_j()
        t0 = time.perf_counter()
        with torch.no_grad():
            gen = quadron.generate(**inputs, max_new_tokens=128)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        energy_j = max(_energy_reading_j() - e_before, 0.0)
        out_txt = tokenizer.decode(gen[0], skip_special_tokens=True)

        bleu = compute_bleu([out_txt], [ref_txt])
        stats.append(ExampleStats(latency_ms, energy_j, bleu))

    # ------------------------------------------------------------------
    # Aggregate & persist
    # ------------------------------------------------------------------
    latencies = np.array([s.latency_ms for s in stats])
    energies = np.array([s.energy_j for s in stats])
    bleus = np.array([s.quality for s in stats])

    results: Dict[str, float] = {
        "latency_ms_mean": float(latencies.mean()),
        "latency_ms_std": float(latencies.std()),
        "energy_j_mean": float(energies.mean()),
        "bleu_mean": float(bleus.mean()),
    }

    out_file = RESULT_DIR / f"{exp_name}.json"
    with out_file.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    # ------------------------------------------------------------------
    # User-visible console output & histogram figure
    # ------------------------------------------------------------------
    print("================  EXPERIMENT 1  ================")
    print("Full description: Latency / Energy / Quality on WMT14 En→De")
    print(json.dumps(results, indent=2))

    sns.set_theme()
    plt.figure(figsize=(6, 4))
    plt.title("Latency distribution – QuADRoN-DM (ms)")
    sns.histplot(latencies, bins=30, kde=True)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Count")
    plt.tight_layout()
    fig_path = IMAGES_DIR / "latency_text.pdf"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()

    return results
