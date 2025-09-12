# src/evaluate.py
"""Evaluation utilities and plotting helpers."""
from __future__ import annotations

import json
import pathlib
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
import torch

sns.set_theme(style="whitegrid")

# -----------------------------------------------------------------------------
#   Paths – mandatory locations (see instructions)
# -----------------------------------------------------------------------------

_RESULTS = pathlib.Path(".research/iteration7/images")
_RESULTS.mkdir(parents=True, exist_ok=True)

__all__ = [
    "evaluate_task",
    "line_plot",
]


def evaluate_task(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    task_id: int,
) -> float:
    """Compute classification accuracy for a single task."""
    model.eval()
    correct = 0
    total = 0
    loader = torch.utils.data.DataLoader(dataset, batch_size=256)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x, task_id).argmax(1).cpu()
            correct += (preds == y).sum().item()
            total += y.numel()
    model.train()
    return 100.0 * correct / total if total else 0.0


# -----------------------------------------------------------------------------
#   Plotting helpers
# -----------------------------------------------------------------------------

def line_plot(
    json_path: pathlib.Path,
    key: str,
    title: str,
    pdf_name: str,
) -> str:
    """Generate a simple line plot from the metrics JSON and save as PDF."""
    data = json.loads(json_path.read_text())
    y: List[float] = data[key]
    x = list(range(len(y)))

    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o", label=key)
    for xi, yi in zip(x, y):
        plt.text(xi, yi, f"{yi:.2f}")
    plt.xlabel("Task")
    plt.ylabel(key)
    plt.title(title)
    plt.legend()

    out_path = _RESULTS / pdf_name
    plt.savefig(out_path, bbox_inches="tight", format="pdf")
    plt.close()
    return str(out_path)
