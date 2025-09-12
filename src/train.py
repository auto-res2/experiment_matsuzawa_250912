"""
train.py
~~~~~~~~
All model architectures and training-related logic live here.  Nothing in this
file performs I/O so that it can be safely imported without side-effects.
"""
from __future__ import annotations

from typing import Any

import torch
import dgl
from dgl.nn import GraphConv


# ---------------------------------------------------------------------------
#  Helper blocks
# ---------------------------------------------------------------------------
class GCN(torch.nn.Module):
    """Very small GCN baseline that is good enough for the synthetic smoke test.
    A real experiment would swap this class for a deeper architecture but keep
    the public API identical so that the driver code in *src.main* requires no
    changes. """

    def __init__(self, in_dim: int, hid_dim: int, n_classes: int, n_layers: int = 2):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GraphConv(in_dim, hid_dim))
        for _ in range(n_layers - 2):
            self.layers.append(GraphConv(hid_dim, hid_dim))
        self.layers.append(GraphConv(hid_dim, n_classes))

    # ---------------------------------------------------------------------
    def forward(self, g: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        h = feat
        for layer in self.layers[:-1]:
            h = torch.relu(layer(g, h))
        return self.layers[-1](g, h)


# ---------------------------------------------------------------------------
#  STARLING-Duo Skeleton (public interface only)
# ---------------------------------------------------------------------------
class StarlingDuo(torch.nn.Module):
    """Skeleton that exposes exactly the methods needed by the experiment
    pipeline – namely *pretrain*, *adapt* and *predict*.  The internals can be
    swapped out later without affecting the rest of the repository. """

    def __init__(self, in_dim: int, hid_dim: int, n_classes: int, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.gnn = GCN(in_dim, hid_dim, n_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    def forward(self, g: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        return self.gnn(g, feat)

    # ------------------------------------------------------------------
    def _loop(
        self,
        g: dgl.DGLGraph,
        feat: torch.Tensor,
        label: torch.Tensor,
        optim: torch.optim.Optimizer,
        epochs: int,
    ) -> None:
        self.train()
        for _ in range(epochs):
            logits = self(g, feat)
            loss = self.loss_fn(logits, label)
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------
    def pretrain(self, g: dgl.DGLGraph, feat: torch.Tensor, label: torch.Tensor):
        optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        epochs = int(self.cfg["general"].get("epochs_pretrain", 5))
        self._loop(g, feat, label, optim, epochs)

    def adapt(self, g: dgl.DGLGraph, feat: torch.Tensor, label: torch.Tensor):
        optim = torch.optim.SGD(self.parameters(), lr=1e-2)
        epochs = int(self.cfg["general"].get("epochs_finetune", 5))
        self._loop(g, feat, label, optim, epochs)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, g: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self(g, feat).argmax(dim=-1)
