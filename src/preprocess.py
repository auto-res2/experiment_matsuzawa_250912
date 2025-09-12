# src/preprocess.py
"""Data-loading and synthetic-graph generation utilities.

Only a minimal subset is implemented so that smoke tests run without any
external data downloads.  If the URI starts with `synthetic://`, we will
create a deterministic Erdős–Rényi graph with the requested node / edge
counts and random binary labels.
"""
from __future__ import annotations

import hashlib
from typing import List

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def _deterministic_seed(string: str) -> int:
    """Convert an arbitrary string to a deterministic integer seed."""
    return int(hashlib.sha256(string.encode()).hexdigest(), 16) % (2**32)


def _make_synthetic_graph(num_nodes: int, num_edges: int, seed: int) -> Data:
    torch.manual_seed(seed)

    # Random edges (avoid self-loops duplicated)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # 16-d node features
    x = torch.randn(num_nodes, 16)

    # Binary labels (balanced)
    y = torch.randint(0, 2, (num_nodes,), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)


def _from_uri(uri: str, window_nodes: int, window_edges: int):  # noqa: D401
    if uri.startswith("synthetic://"):
        seed = _deterministic_seed(uri)
        graph = _make_synthetic_graph(window_nodes, window_edges, seed)
        return [graph]

    raise RuntimeError(
        "URI scheme not supported and no real dataset loader implemented.  "
        "STRICT NO-FALLBACK – please supply a supported URI."
    )


# --------------------------------------------------------------------------
# Public helper
# --------------------------------------------------------------------------

def load_graphs(dataset_cfg) -> List[Data]:
    """Load graphs according to the provided DatasetCfg object."""

    return _from_uri(dataset_cfg.uri, dataset_cfg.window_nodes, dataset_cfg.window_edges)


def make_dataloaders(graphs: List[Data], batch_edges: int):
    """Return a DataLoader that iterates over full graphs.

    We keep it simple: one graph per batch (GraphSAINT etc. is out of scope
    for smoke tests).  The *batch_edges* argument is therefore ignored but
    kept to satisfy the TrainCfg API.
    """

    return DataLoader(graphs, batch_size=1, shuffle=True)
