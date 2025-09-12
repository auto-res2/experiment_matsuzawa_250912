# src/evaluate.py
"""Model evaluation and plotting utilities."""
from __future__ import annotations

from typing import List

import matplotlib
import matplotlib.pyplot as plt
import torch

matplotlib.use("Agg")  # head-less backend – we only write files

__all__: List[str] = ["evaluate", "plot_metrics"]


def evaluate(model, data, device):
    """Top-1 accuracy on the boolean ``test_mask`` of the given PyG ``Data``."""
    model.eval()
    with torch.inference_mode():
        pred = model(data.x.to(device), data.edge_index.to(device)).softmax(dim=-1)
        y_true = data.y.to(device)
        y_pred = pred.argmax(dim=-1)
        correct = int((y_pred[data.test_mask] == y_true[data.test_mask]).sum())
        acc = correct / int(data.test_mask.sum())
    return acc


def plot_metrics(
    train_losses: List[float],
    test_accs: List[float],
    out_file,
    title: str = "Training Loss & Test Accuracy",
):
    """Two-axis plot: loss (left, red) and accuracy (right, blue)."""

    epochs = range(1, len(train_losses) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 4))

    # Loss (left-axis)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:red")
    ax1.plot(epochs, train_losses, color="tab:red", label="Training Loss")
    for x, y in zip(epochs, train_losses):
        ax1.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6)

    # Accuracy (right-axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Test Accuracy", color="tab:blue")
    ax2.plot(epochs, test_accs, color="tab:blue", label="Test Accuracy")
    for x, y in zip(epochs, test_accs):
        ax2.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6)

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.title(title)
    plt.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
