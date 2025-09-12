# src/train.py
"""
Model construction and adaptive precision / width control.
This file is a direct extraction from the original single-file script – no
behavioural changes besides a few robustness patches that were required to keep
runtime stable after the refactor (see comments).
"""
from __future__ import annotations

import os
from typing import Any, Tuple

import torch
from torch import nn

# low-precision kernels (BitsAndBytes 4-/8-bit linear layers)
import bitsandbytes as bnb  # noqa: E402

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

HF_TOKEN_ENV = "HF_TOKEN"  # name of env-variable that stores an access token


class SuperSurrogate(nn.Module):
    """Width- & precision-elastic wrapper around a HF seq-to-seq model."""

    def __init__(self, base_model_name: str, hf_token: str | None = None):
        super().__init__()

        auth_kwargs: dict[str, Any] = {"use_auth_token": hf_token} if hf_token else {}
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, **auth_kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, **auth_kwargs)
        except Exception as exc:  # pragma: no cover – network / HF issues
            raise RuntimeError(
                f"Unable to download or load model '{base_model_name}'. Aborting.\n{exc}"
            ) from exc

        # internal state – will be updated by the public setters below
        self._current_bit: int = 16
        self._current_width: float = 1.0

        # start in half precision (fp16) for a decent speed / memory baseline
        self.model.half()

    # ------------------------------------------------------------------
    # public knobs
    # ------------------------------------------------------------------
    def set_precision(self, bits: int) -> None:
        """Switch between 4/8/16-bit weights at runtime."""
        if bits not in {4, 8, 16}:
            raise ValueError("bits must be one of {4, 8, 16}")
        if bits == self._current_bit:
            return  # already at requested bit-depth

        if bits == 16:
            self.model.float()  # reload parameters to fp32/fp16 (depending on dtype)
        else:
            # replace Linear layers by their 4-/8-bit counterparts from bitsandbytes
            for long_name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    quant_cls = bnb.nn.Linear4bit if bits == 4 else bnb.nn.Linear8bitLt
                    new_mod = quant_cls(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                    )
                    # copy weights / bias
                    new_mod.weight.data = module.weight.data.clone()
                    if module.bias is not None:
                        new_mod.bias.data = module.bias.data.clone()

                    parent, child_name = self._find_parent(long_name)
                    setattr(parent, child_name, new_mod)
        self._current_bit = bits

    def set_width(self, ratio: float) -> None:
        """Naive structured pruning: keep top-`ratio` channels based on L1 norm."""
        if not (0.0 < ratio <= 1.0):
            raise ValueError("width ratio must be in (0, 1]")
        if abs(ratio - self._current_width) < 1e-3:
            return  # unchanged

        for _, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                keep = int(module.out_features * ratio)
                if keep < 1:
                    raise ValueError("width ratio too small – no neurons left after pruning")
                # importance = L1 norm across in_features
                importance = module.weight.abs().sum(dim=1)
                topk = torch.topk(importance, keep).indices
                mask = torch.zeros_like(importance, dtype=torch.bool)
                mask[topk] = True

                module.weight.data = module.weight.data[mask]
                if module.bias is not None:
                    module.bias.data = module.bias.data[mask]
                module.out_features = keep
        self._current_width = ratio

    # ------------------------------------------------------------------
    # generation / routing
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def forward(self, text: str, *, router_temperature: float = 1.0, max_new_tokens: int = 60) -> str:  # noqa: D401,E501
        """Translate *text* ⇒ German using a toy entropy-based router."""
        toks = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        # --- toy routing decision (very cheap) ---
        logits = self.model.generate(
            **toks, max_new_tokens=0, return_dict_in_generate=True
        ).scores[0]
        ent = torch.distributions.Categorical(logits=logits / router_temperature).entropy()
        if ent.mean() > 4.0:  # ambiguous → take full-precision, full-width path
            self.set_precision(16)
            self.set_width(1.0)

        out_ids = self.model.generate(**toks, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _find_parent(self, long_name: str) -> Tuple[nn.Module, str]:
        """Return the parent module and attribute-name of *long_name*.

        Example: long_name == "encoder.layers.0.self_attn.q_proj".
        """
        components = long_name.split(".")
        parent = self.model  # start from root
        for comp in components[:-1]:
            parent = getattr(parent, comp)
        return parent, components[-1]


# ----------------------------------------------------------------------
# convenience factory (keeps train.py self-contained)
# ----------------------------------------------------------------------

def build_super_surrogate(model_name: str) -> SuperSurrogate:
    """Create a `SuperSurrogate` and move it to an available device."""
    hf_token = os.getenv(HF_TOKEN_ENV)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SuperSurrogate(model_name, hf_token=hf_token).to(device)
