"""src/evaluate.py
Basic evaluation utilities that compute loss & accuracy on a supplied
validation set.  This replaces the previous *NotImplemented* placeholder so
that the experiment produces a concrete numerical JSON artefact.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def evaluate_model(  # noqa: D401
    *,
    model: torch.nn.Module,
    val_loader: DataLoader,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run a forward pass over *val_loader* and compute loss & accuracy."""

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            running_loss += loss.item() * xb.size(0)
            correct += (out.argmax(dim=1) == yb).sum().item()
            total += xb.size(0)

    if total == 0:  # pragma: no cover – protects against empty loader
        raise ValueError("Validation loader is empty – cannot compute metrics.")

    results = {
        "val_loss": float(running_loss / total),
        "val_accuracy": float(correct / total),
        "num_samples": int(total),
    }

    logger.info("Evaluation – loss: %.4f  acc: %.4f", results["val_loss"], results["val_accuracy"])
    return results
