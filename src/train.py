"""src/train.py
Training-related utilities.
Currently, the original single-file experiment did not yet include any actual
model-training code – only dataset / configuration validation logic.  To keep
the refactor 100 % faithful to the provided source (STRICT NO-FALLBACK RULE),
we merely expose minimal stubs so that future extensions can plug real models
here without changing any import paths.
"""
from __future__ import annotations

from typing import Dict, Any

__all__ = [
    "run_training_pipeline",
]

def run_training_pipeline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Dummy training placeholder.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Full experiment configuration as loaded from YAML.

    Returns
    -------
    Dict[str, Any]
        A dictionary mimicking training metrics.  Since the reference
        implementation has no real model, we only return a stub that allows
        the caller to write a JSON result file without crashing.
    """
    # NOTE: do *not* implement real training here – out of scope for the given
    # experiment scaffold.
    return {
        "train_status": "skipped (no model code provided in reference script)",
        "cfg_hash": hash(str(cfg)) & 0xFFFFFFFF,
    }
