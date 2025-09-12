"""src/preprocess.py
Data-loading and preprocessing utilities extracted from the original experiment should be placed here.
Because no experiment code was provided, the functions only raise NotImplementedError.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)


def load_and_preprocess_data(config: Dict[str, Any]) -> Tuple[Any, Any]:  # noqa: D401
    """Load and preprocess data according to configuration.

    Parameters
    ----------
    config : Dict[str, Any]
        Parsed YAML configuration.

    Returns
    -------
    Tuple[Any, Any]
        Training and validation datasets (or dataloaders).
    """
    raise NotImplementedError("Data-loading logic must be inserted here from the original script.")
