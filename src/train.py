"""
train.py
===========
All training-specific utilities are collected here so that they can be
re-used by every experiment.  Nothing in this file performs heavy work on
import – that is handled by `src.main`.
"""
from __future__ import annotations

from typing import List

import torch
from torch import nn
import tqdm

# ---------------------------------------------------------------------------
# Public symbols
# ---------------------------------------------------------------------------

__all__: List[str] = [
    "train_one_epoch",
    "evaluate",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy (private helper; exposed via evaluate/evaluate).

    Parameters
    ----------
    logits : torch.Tensor
        Model outputs of shape ``[B, C]``.
    labels : torch.Tensor
        Integer class indices of shape ``[B]``.
    """
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimiser: torch.optim.Optimizer,
    device: torch.device | str,
) -> float:
    """Train ``model`` for a single epoch.

    Returns
    -------
    float
        Average *training* accuracy (coarse sanity check only).  Proper
        validation must be performed outside this function.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_acc, seen = 0.0, 0

    for batch in tqdm.tqdm(dataloader, desc="train", leave=False):
        imgs = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimiser.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimiser.step()

        total_acc += _accuracy(logits.detach(), labels) * imgs.size(0)
        seen += imgs.size(0)

    return total_acc / max(seen, 1)


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device | str,
) -> float:
    """Validation helper returning top-1 accuracy."""
    model.eval()
    total_acc, seen = 0.0, 0

    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(imgs)
            total_acc += _accuracy(logits, labels) * imgs.size(0)
            seen += imgs.size(0)

    return total_acc / max(seen, 1)
