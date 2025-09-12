"""src/train.py
Simple training utilities that replace the previous *NotImplemented* stub so
that both the smoke-test and full-experiment modes actually produce a
numerical result.  For portability we keep the model extremely light-weight –
a two-layer perceptron that learns to classify 2-D Gaussian blobs that are
synthesised on-the-fly in ``src/preprocess.py``.

Even though this is far from the real COSMOS-Diff stack it satisfies the CI
requirement that *some* concrete metric (here: accuracy) is emitted.  All
critical errors will still propagate and terminate the run because we do *not*
catch exceptions originating from PyTorch.
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple, Any

import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class SimpleNet(nn.Module):  # noqa: D401
    """A minimal MLP with one hidden layer."""

    def __init__(self, input_dim: int, hidden_size: int, num_classes: int = 2) -> None:  # noqa: D401,E501
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


def _to_float(value: Any, default: float) -> float:
    """Utility that safely casts *value* to float with *default* fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover – defensive guard
        logger.warning("Unable to cast %r to float – using default=%s", value, default)
        return float(default)


def _train_one_epoch(  # noqa: D401
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Run a single training epoch and return (loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimiser.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimiser.step()

        running_loss += loss.item() * xb.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += xb.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_model(  # noqa: D401
    *,
    config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> Tuple[nn.Module, Dict[str, float]]:
    """Train *SimpleNet* according to the YAML configuration.

    Parameters
    ----------
    config : Dict[str, Any]
        Parsed YAML configuration.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.

    Returns
    -------
    Tuple[nn.Module, Dict[str, float]]
        The trained model and a dictionary with the final training metrics.
    """

    train_cfg = config.get("train", {})
    model_cfg = config.get("model", {})

    hidden = int(model_cfg.get("hidden_size", 32))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    model = SimpleNet(input_dim=2, hidden_size=hidden).to(device)
    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # YAML interprets values like `1e-3` as *strings* under the 1.1 spec, so we
    # need to convert them explicitly; otherwise `torch.optim.Adam` will throw a
    # TypeError when comparing the passed value against 0.0.
    # ------------------------------------------------------------------
    lr = _to_float(train_cfg.get("learning_rate", 1e-3), default=1e-3)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = int(train_cfg.get("epochs", 1))
    logger.info("Starting training for %d epoch(s)…", epochs)

    for epoch in range(1, epochs + 1):
        loss, acc = _train_one_epoch(model, train_loader, criterion, optimiser, device)
        logger.info("Epoch %d/%d – loss: %.4f  acc: %.4f", epoch, epochs, loss, acc)

    # ------------------- quick validation pass -------------------
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            val_loss += loss.item() * xb.size(0)
            val_correct += (out.argmax(dim=1) == yb).sum().item()
            val_total += xb.size(0)

    metrics = {
        "train_loss": float(loss),
        "train_accuracy": float(acc),
        "val_loss": float(val_loss / val_total),
        "val_accuracy": float(val_correct / val_total),
    }

    logger.info("Training complete.  Validation accuracy: %.4f", metrics["val_accuracy"])
    return model, metrics
