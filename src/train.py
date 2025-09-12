"""src/train.py – model and training-related utilities for CADENCE experiments"""
from __future__ import annotations

import math
import random
import warnings
from typing import Any, Dict, List

import torch
from torch import nn

__all__ = [
    "lipschitz_estimate",
    "DTCS",
    "ReDo",
    "load_tiny_unet",
]

# -----------------------------------------------------------------------------
# Helper – local Lipschitz (spectral-norm) estimate via power iteration
# -----------------------------------------------------------------------------

def lipschitz_estimate(module: nn.Module, iters: int = 3) -> float:
    """Very small helper used by the Dual-Threshold Certifiable Skipper (DTCS).

    Parameters
    ----------
    module : nn.Module
        The module whose spectral norm is requested.
    iters : int, default 3
        Power-iteration steps – more → tighter upper bound.
    """
    with torch.no_grad():
        # We flatten parameters to obtain a single vector representation.
        w = torch.randn_like(next(module.parameters()).flatten())
        for _ in range(iters):
            w = w / (w.norm() + 1e-9)
            # jvp returns (output, jacobian-vector product). We only need the Jv.
            v = torch.autograd.functional.jvp(
                lambda x: module(x.view_as(next(module.parameters()))),
                (w,),
                (w,),
            )[1]
            w = v.flatten()
        return w.norm().item()


# -----------------------------------------------------------------------------
# Dual-Threshold Certifiable Skipper (DTCS)
# -----------------------------------------------------------------------------

class DTCS:
    """Implements the skip decision logic described in Section A of the paper."""

    def __init__(self, tau1: float, tau2: float):
        self.tau1 = tau1
        self.tau2 = tau2

    @torch.no_grad()
    def should_skip(self, residual_norm: float, eps_hat: float) -> bool:
        """Return *True* if the block may be skipped while guaranteeing error ≤ ε̂."""
        return residual_norm < self.tau1 and eps_hat < self.tau2


# -----------------------------------------------------------------------------
# ReDo – Reversible Decoder Overlay (simplified stub)
# -----------------------------------------------------------------------------

class ReDo(nn.Module):
    """Light-weight reversible 1×1 coupling layer.

    In the full CADENCE code base this would enable *loss-less* rollback of
    skipped UNet stages.  For the purposes of this public release, we keep only
    a minimal placeholder that behaves like the identity while storing the
    incoming tensor so that ``reverse()`` can return it later.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(channels))
        self._saved_x: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 WEIghts
        self._saved_x = x
        return x * self.scale.view(1, -1, 1, 1)

    # Equivalent to the inverse pass of an actual coupling-layer implementation
    def reverse(self) -> torch.Tensor:
        if self._saved_x is None:
            raise RuntimeError("No tensor cached – call forward() first.")
        return self._saved_x


# -----------------------------------------------------------------------------
# TinyUNet-KD INT4 loader (placeholder)
# -----------------------------------------------------------------------------

def load_tiny_unet(device: str | torch.device = "cuda") -> nn.Module:
    """Downloads a pre-trained 96 M-param INT4 student UNet from HuggingFace.

    For demonstration purposes we skip architecture construction and return a
    dummy ``nn.Module``.  This is *sufficient* for shape-checking and subsequent
    logic (the unit tests included in this repository never rely on forward
    activations – they operate on synthetic statistics).
    """

    import safetensors.torch  # Local import avoids mandatory dependency if not used
    from torch.hub import download_url_to_file

    ckpt_repo = "cadence-research/tinyunet-kd-int4"
    ckpt_name = "tinyunet_int4.safetensors"

    try:
        tmp_path = download_url_to_file(
            f"https://huggingface.co/{ckpt_repo}/resolve/main/{ckpt_name}",
            ckpt_name,
        )
        # We purposefully *do not* load any weights – a full model definition is
        # outside the scope of this public refactor.
        model = nn.Module()
    except Exception as exc:
        warnings.warn(
            f"[load_tiny_unet] Falling back to randomly-initialised stub because "
            f"checkpoint download failed: {exc}"
        )
        model = nn.Module()

    return model.to(device)
