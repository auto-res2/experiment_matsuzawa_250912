"""
train.py
===========
All training-specific utilities are collected here so that they can be
re-used by every experiment.  Nothing in this file should perform any
expensive work when it is imported – that is handled by `src.main`.
"""
from __future__ import annotations

import torch
from torch import nn
import tqdm
from typing import Dict


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy helper (private; exported via evaluate.py)."""
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimiser: torch.optim.Optimizer,
    device: torch.device | str,
) -> float:
    """Standard supervised training loop for one epoch.

    Returns
    -------
    float
        Average accuracy computed on the *training* batches (purely for a
        coarse sanity-check; proper validation happens outside).
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
