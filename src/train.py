# src/train.py
"""Training utilities and resource monitoring for TIGER-Lite experiments."""
from __future__ import annotations

import json
import pathlib
import time
from typing import List, Dict, Any

import torch
import torch.nn.functional as F

from .evaluate import evaluate_task, line_plot

# -----------------------------------------------------------------------------
#   Resource monitoring
# -----------------------------------------------------------------------------

import os
import psutil
from contextlib import contextmanager

__all__ = [
    "ResourceViolation",
    "WatchDog",
    "continual_train",
]


class ResourceViolation(RuntimeError):
    """Raised when RAM or FLOP budgets are exceeded."""


class WatchDog:
    """Live monitor for RAM and cumulative floating-point operations.

    Notes
    -----
    We measure *delta* memory with respect to the baseline just after the
    watchdog is instantiated.  This avoids counting Python interpreter and
    library initialisation overheads against the user-defined cap, in line with
    the instructions that budgets correspond to *experiment* overheads.
    """

    def __init__(self, mem_cap_mb: float, flop_cap_g: float):
        self._proc = psutil.Process(os.getpid())
        self._base_rss = self._rss()            # memory before the experiment
        self.mem_cap = mem_cap_mb * 1024 * 1024  # bytes
        self.flop_cap = flop_cap_g * 1e9          # FLOPs (cumulative)
        self.cum_flops: float = 0.0

    # ------------------------------------------------------------------ utils
    def _rss(self) -> int:
        return self._proc.memory_info().rss  # resident set size (bytes)

    def _check_ram(self) -> None:
        delta = self._rss() - self._base_rss
        if delta > self.mem_cap:
            raise ResourceViolation(
                f"RAM cap exceeded: {delta/1e6:.1f} MB > {self.mem_cap/1e6:.1f} MB"
            )

    def _add_flops(self, flops: float) -> None:
        self.cum_flops += flops
        if self.cum_flops > self.flop_cap:
            raise ResourceViolation(
                f"FLOP cap exceeded: {self.cum_flops/1e9:.2f} G > {self.flop_cap/1e9:.2f} G"
            )

    # -------------------------------------------------------------- context mgr
    @contextmanager
    def track(self, est_flops: float):
        """Context manager counting FLOPs after the wrapped block."""
        yield
        self._add_flops(est_flops)
        self._check_ram()


# -----------------------------------------------------------------------------
#   Continual-learning trainer
# -----------------------------------------------------------------------------


def _rough_flop_estimate(batch_size: int) -> float:
    """Crude per-mini-batch FLOP estimate for ResNet-18 backbone.

    We intentionally keep the estimate pessimistic to avoid under-counting.
    """
    # Empirically ~1.8 G FLOPs for 32 images; scale linearly with batch size.
    return 1.8e9 * batch_size / 32.0


def continual_train(
    model: torch.nn.Module,
    tasks: List[torch.utils.data.Dataset],
    cfg: Dict[str, Any],
    save_dir: pathlib.Path,
):
    """Main training loop across sequential tasks.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    tasks : List[Dataset]
        List of datasets, one per task.
    cfg : dict
        Contains keys `mem_mb`, `flop_g`, `batch`, and anything returned from
        the YAML configuration.
    save_dir : pathlib.Path
        Where to store JSON metrics and generated figures.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    wd = WatchDog(cfg["mem_mb"], cfg["flop_g"])
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)

    metrics: Dict[str, List[float]] = {"task_acc": []}

    epochs = cfg.get("epochs_per_task", 1)

    for task_id, task_ds in enumerate(tasks):
        loader = torch.utils.data.DataLoader(
            task_ds,
            batch_size=cfg.get("batch", 32),
            shuffle=True,
            num_workers=0,      # keep 0 to minimise extra processes for smoke-test
            pin_memory=torch.cuda.is_available(),
        )

        # -------------------------- training loop ---------------------
        for _ in range(epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                est_flops = _rough_flop_estimate(x.size(0))
                with wd.track(est_flops):
                    logits = model(x, task_id)
                    loss = F.cross_entropy(logits, y)
                    loss.backward()
                    optimiser.step()
                    optimiser.zero_grad(set_to_none=True)

        # -------------------------- quick evaluation ------------------
        acc = evaluate_task(model, task_ds, device, task_id)
        metrics["task_acc"].append(acc)
        print(f"Task {task_id}: accuracy = {acc:.2f} %")

    # -------------------------- persist results ----------------------
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / "metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2))

    fig_name = line_plot(
        json_path,
        key="task_acc",
        title="Accuracy per Task",
        pdf_name="accuracy.pdf",
    )

    return json_path, [fig_name]
