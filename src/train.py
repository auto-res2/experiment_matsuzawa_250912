"""src/train.py
Model-related helpers: loading teacher/candidate sequence models and diffusion
pipelines.  All logic comes directly from the original single-file script; only
minor device/precision guards were added to avoid crashes on CPU–only
machines.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import torch
from diffusers import DiffusionPipeline
from transformers import AutoModelForSeq2SeqLM

__all__ = [
    "load_hf_model",
    "load_diffusion",
]


def _device() -> str:  # small helper to keep the rest of the code tidy
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_hf_model(model_name: str) -> AutoModelForSeq2SeqLM:
    """Load an HF encoder-decoder model, die early if inaccessible."""
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        model.eval()
        return model.to(_device())
    except Exception as exc:
        sys.exit(
            f"[ERROR] Could not load model '{model_name}'. Make sure the model is "
            f"public or supply a valid HF_TOKEN. Original error: {exc}"
        )


def load_diffusion(model_name: str) -> DiffusionPipeline:
    """Load a diffusion pipeline with optional auth token."""
    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            token=os.getenv("HF_TOKEN"),
        )
        return pipe.to(_device())
    except Exception as exc:
        sys.exit(f"[ERROR] Could not load diffusion model '{model_name}'. {exc}")
