"""
train.py – training-/experiment–orchestrator logic extracted from the original monolithic
script.  All heavy-lifting model code is still assumed to live inside the (un-shipped)
`refuse_cl` package exactly as in the original repository.  This file therefore keeps the
same dispatching behaviour and the strict fail-fast semantics for missing artefacts.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

import torch  # noqa: F401 – referenced by downstream packages (device checks, etc.)
from huggingface_hub import hf_hub_download

from .preprocess import fetch_dataset

logger = logging.getLogger("ReFuse-CL")

ROOT = Path(__file__).resolve().parent.parent  # project root (…/src/..)

__all__ = [
    "run_experiment",
]

# -----------------------------------------------------------------------------
#  Pre-load HF backbones (fail-fast behaviour preserved from original script)
# -----------------------------------------------------------------------------

def _preload_backbones(cfg: Dict[str, Any]) -> None:
    """Download the three backbone models referenced in the configuration.

    The function intentionally does *not* keep the models in memory – it only
    makes sure the artefacts are present in the local cache so that subsequent
    training runs will not hit the network any more.  If any of the downloads
    fail we abort immediately in accordance with the STRICT NO-FALLBACK rule
    laid out in the original paper’s reproducibility checklist.
    """
    logger.info("Pre-loading backbone checkpoints from HuggingFace Hub …")
    vision_id = cfg["model"]["vision_backbone"]
    text_id = cfg["model"]["text_backbone"]
    audio_id = cfg["model"]["audio_backbone"]

    for hf_id in [vision_id, text_id, audio_id]:
        try:
            logger.info("  ↳ %s", hf_id)
            hf_hub_download(
                repo_id=hf_id,
                filename="pytorch_model.bin",
                cache_dir=str(ROOT / ".hf_cache"),
            )
        except Exception as err:
            raise RuntimeError(
                f"Unable to download required model artefact '{hf_id}' from the HuggingFace Hub. "
                f"Strict NO-FALLBACK rule prohibits synthetic substitutes. Error: {err}"
            ) from err


# -----------------------------------------------------------------------------
#  High-level experiment dispatcher (verbatim from original script)
# -----------------------------------------------------------------------------


def run_experiment(cfg: Dict[str, Any]) -> None:
    """Dispatch to Exp-1 / Exp-2 / Exp-3 as configured.

    Heavy-weight implementation details live in `refuse_cl.experiments.*` which
    are *not* part of this refactor.  Import errors therefore remain a feature,
    not a bug – the reference implementation has to be available in the Python
    path to run the full benchmark.
    """
    exp_id = cfg.get("experiment_id", "exp1").lower()
    logger.info("===============  Running %s  ===============", exp_id.upper())

    # 1) Acquire / prepare data (common to all experiments)
    data_root = fetch_dataset(cfg)

    # 2) Make sure HF backbones are locally cached so we fail fast if offline
    _preload_backbones(cfg)

    # 3) Route execution -------------------------------------------------------
    if exp_id == "exp1":
        try:
            from refuse_cl.experiments.exp1_end2end import run_exp1
        except ModuleNotFoundError as err:  # pragma: no cover – optional dependency
            raise ModuleNotFoundError(
                "`refuse_cl` package with experiment implementation missing. "
                "Install it to run the full benchmark."
            ) from err
        run_exp1(cfg, data_root)

    elif exp_id == "exp2":
        try:
            from refuse_cl.experiments.exp2_unlearning import run_exp2
        except ModuleNotFoundError as err:  # pragma: no cover
            raise ModuleNotFoundError(
                "`refuse_cl` package with experiment implementation missing."
            ) from err
        run_exp2(cfg, data_root)

    elif exp_id == "exp3":
        try:
            from refuse_cl.experiments.exp3_flash_carbon import run_exp3
        except ModuleNotFoundError as err:  # pragma: no cover
            raise ModuleNotFoundError(
                "`refuse_cl` package with experiment implementation missing."
            ) from err
        run_exp3(cfg, data_root)

    else:
        raise ValueError(f"Unknown experiment_id '{exp_id}' in configuration.")
