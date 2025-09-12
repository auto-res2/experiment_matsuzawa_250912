# src/train.py
"""Model architectures and training utilities for the CELESTE refactored
experiments.  All heavy-weight logic was already implemented in the original
single-file script – this module simply groups the relevant code so that it can
be imported from the orchestrating ``src.main`` entry-point.
"""
from __future__ import annotations

from typing import List

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

try:
    from torch_geometric.nn import SAGEConv
except ImportError as e:  # pragma: no cover – enforced at runtime
    import sys

    sys.stderr.write(
        "[FATAL] PyTorch-Geometric not found – install a build that matches your CUDA / PyTorch versions.\n"
    )
    sys.exit(2)

__all__: List[str] = [
    "CelesteAdapter",
    "CelesteGNN",
    "train_one_epoch",
]


class CelesteAdapter(nn.Module):
    """A tiny spectral adapter used for cold-start nodes in the full paper.

    In the condensed demo we keep a single linear projection followed by ReLU.
    """

    def __init__(self, in_dim: int, out_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.proj(x))


class CelesteGNN(nn.Module):
    """Minimal GraphSAGE backbone + adapter.

    The real project would import the heavy-weight implementation from an
    external package; here we embed a light version so that the public test
    environment can run end-to-end in <30 s.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        hidden: int = 256,
        num_layers: int = 6,
        num_classes: int = 47,
    ) -> None:
        super().__init__()
        self.adapter = CelesteAdapter(in_channels, hidden)
        self.convs = nn.ModuleList([SAGEConv(hidden, hidden) for _ in range(num_layers)])
        self.final = nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index):
        h = self.adapter(x)
        for conv in self.convs:
            h = conv(h, edge_index).relu()
        return self.final(h)


# -----------------------------------------------------------------------------
# Training loop (single epoch) – identical to the original implementation
# -----------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,  # noqa: D401 – single-object iterable in the demo
    criterion: nn.Module,
    optimiser: optim.Optimizer,
    device: torch.device,
) -> float:
    """One stochastic/full batch epoch – returns the mean loss."""

    model.train()
    running_loss = 0.0

    for batch in loader:  # in the demo ``loader`` is ``[data]``
        optimiser.zero_grad(set_to_none=True)
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        optimiser.step()
        running_loss += loss.item() * batch.num_graphs

    return running_loss / len(loader.dataset)
