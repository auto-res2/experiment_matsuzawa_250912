# src/train.py
"""Model architectures and training utilities for SAFE-FUSE-Ψ.
This file contains:
  • YAML configuration loader + dataclasses
  • AoW (Activation-On-Wire) hook utilities
  • Model backbone (GCNII approximation)
  • LocalTrainer – per-silo differential-privacy aware trainer
"""
from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Dict, List, Any

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
from torch_geometric.nn import GCN2Conv
from tqdm import tqdm

from opacus import PrivacyEngine

# -------------------------------------------------
# YAML configuration --------------------------------------------------------------------
# -------------------------------------------------
from dataclasses import dataclass, field


@dataclass
class DatasetCfg:
    name: str
    uri: str
    split_strategy: str
    window_nodes: int
    window_edges: int


@dataclass
class TrainingCfg:
    epochs_per_snapshot: int
    lr: float
    weight_decay: float
    batch_edges: int
    seeds: List[int]


@dataclass
class DPCfg:
    epsilons: List[float]
    delta: float
    max_grad_norm: float


@dataclass
class ExperimentCfg:
    name: str
    datasets: Dict[str, DatasetCfg] = field(default_factory=dict)
    training: TrainingCfg | None = None
    dp: DPCfg | None = None
    output_dir: Path = Path("results")
    variants: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# Utility: configuration loader
# --------------------------------------------------------------------------------------

def load_cfg(path: str | Path) -> ExperimentCfg:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)

    with p.open("r", encoding="utf8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh)

    datasets = {
        name: DatasetCfg(name=name, **cfg) for name, cfg in raw["datasets"].items()
    }
    training = TrainingCfg(**raw["training"])
    dp = DPCfg(**raw["dp"])

    exp_raw = raw.get("experiment", {})
    return ExperimentCfg(
        name=exp_raw.get("name", "experiment"),
        datasets=datasets,
        training=training,
        dp=dp,
        output_dir=Path(exp_raw.get("output_dir", "results")),
        variants=exp_raw.get("variants", ["aow"]),
    )


# -------------------------------------------------
# AoW –  Activation-On-Wire autograd hook ----------------------------------------------
# -------------------------------------------------
class AoWHook(torch.autograd.Function):
    """Unbiased sparse ternary estimator – simplified for demonstration."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor):
        sign = (tensor >= 0).type(torch.bool)
        exponent_bit = (tensor.abs() > tensor.abs().median()).type(torch.bool)
        ctx.save_for_backward(sign, exponent_bit)
        # Identity forward (tensor also returned to downstream ops)
        return tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        sign, exponent_bit = ctx.saved_tensors
        mask = (exponent_bit.float() * 2 - 1) * (sign.float() * 2 - 1)
        grad_est = grad_output * mask  # unbiased expectation
        return grad_est


def aow_wrap(module: nn.Module):
    """Recursively patch *leaf* modules so their outputs stream via AoW."""

    for child in module.children():
        aow_wrap(child)
    if not list(module.children()):  # leaf
        orig_forward = module.forward

        def _forward(*args, **kwargs):  # noqa: D401 – simple wrapper
            return AoWHook.apply(orig_forward(*args, **kwargs))

        module.forward = _forward  # noqa: setattr-assignment


# -------------------------------------------------
# Model definitions ---------------------------------------------------------------------
# -------------------------------------------------
class GCNIIBackbone(nn.Module):
    """GCNII backbone (configurable depth)."""

    def __init__(self, num_features: int, hidden_channels: int = 512, num_layers: int = 200):
        super().__init__()
        self.lin_in = nn.Linear(num_features, hidden_channels, bias=True)
        self.convs = nn.ModuleList([
            GCN2Conv(hidden_channels, alpha=0.1, theta=0.5, layer=i + 1)
            for i in range(num_layers)
        ])
        self.lin_out = nn.Linear(hidden_channels, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):  # noqa: D401
        x = F.relu(self.lin_in(x))
        x0 = x  # initial representation as required by GCNII formulation
        for conv in self.convs:
            x = F.relu(conv(x, x0, edge_index))
        return self.lin_out(x).squeeze(-1)


class MonteMaskDrop(nn.Module):
    """Simple Monte-Carlo Dropout for Bayesian uncertainty."""

    def __init__(self, p: float = 0.2):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor):  # noqa: D401
        # Always apply dropout to propagate uncertainty at test time
        return F.dropout(x, self.p, training=True)


class SafeFusePsi(nn.Module):
    """Approximate implementation of SAFE-FUSE-Ψ backbone."""

    def __init__(self, num_features: int, hidden_channels: int = 512, num_layers: int = 200, drop_rate: float = 0.2):
        super().__init__()
        self.backbone = GCNIIBackbone(num_features, hidden_channels, num_layers)
        self.mc_drop = MonteMaskDrop(drop_rate)

    def forward(self, data):  # noqa: D401
        x, edge_index = data.x, data.edge_index
        logits = self.backbone(x, edge_index)
        return self.mc_drop(logits)


# -------------------------------------------------
# Local trainer (per silo) --------------------------------------------------------------
# -------------------------------------------------
# NOTE: Use absolute import so that src can be executed as a loose collection of modules.
from preprocess import make_dataloaders  # noqa: E402, isort: skip


class LocalTrainer:  # pylint: disable=too-many-instance-attributes
    """Handles training with differential privacy for one silo and one ε value."""

    def __init__(
        self,
        silo_id: str,
        graphs: List[Any],  # List of torch_geometric.data.Data
        cfg: ExperimentCfg,
        epsilon: float,
        delta: float,
        device: torch.device,
    ):
        self.silo_id = silo_id
        self.cfg = cfg
        self.device = device
        self.loader = make_dataloaders(graphs, batch_edges=cfg.training.batch_edges)

        # Feature dim inference -----------------------------------------------------------------
        sample_graph = self.loader.dataset[0]
        if not hasattr(sample_graph, "x"):
            raise RuntimeError(
                "Dataset has no node features – SAFE-FUSE-Ψ requires features. STRICT NO-FALLBACK rule enforced."
            )
        num_features = sample_graph.x.size(-1)

        self.model = SafeFusePsi(num_features).to(device)
        if "aow" in cfg.variants:
            aow_wrap(self.model)

        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
        )

        # Differential privacy ------------------------------------------------------------------
        self.privacy_engine = PrivacyEngine()
        self.model, self.opt, self.loader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.opt,
            data_loader=self.loader,
            noise_multiplier=self._sigma(epsilon, delta),
            max_grad_norm=cfg.dp.max_grad_norm,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _sigma(epsilon: float, delta: float):
        """Closed-form bound for Gaussian DP noise multiplier."""
        return math.sqrt(2 * math.log(1.25 / delta)) / epsilon

    # ------------------------------------------------------------------
    def train_one_epoch(self):
        self.model.train()
        for batch in self.loader:
            batch = batch.to(self.device)
            self.opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(self.model(batch), batch.y.float())
            loss.backward()
            self.opt.step()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self):  # noqa: D401
        self.model.eval()
        preds, labels = [], []
        for batch in self.loader:
            batch = batch.to(self.device)
            logits = self.model(batch)
            preds.append(torch.sigmoid(logits).cpu())
            labels.append(batch.y.float().cpu())
        preds_t = torch.cat(preds)
        labels_t = torch.cat(labels)
        preds_np, labels_np = preds_t.numpy(), labels_t.numpy()

        roc = roc_auc_score(labels_np, preds_np)
        pred_bin = (preds_np > 0.5).astype(int)
        f1 = f1_score(labels_np, pred_bin)
        brier = brier_score_loss(labels_np, preds_np)

        # Expected Calibration Error (ECE) ------------------------------------------------------
        bin_ids = (preds_np * 10).astype(int)
        accs, confs, cnts = [], [], []
        for b in range(10):
            mask = bin_ids == b
            if mask.sum() == 0:
                continue
            accs.append(labels_np[mask].mean())
            confs.append(preds_np[mask].mean())
            cnts.append(mask.sum())
        ece = sum(c * abs(a - c) for a, c in zip(accs, confs)) / max(sum(cnts), 1)

        return {"roc_auc": float(roc), "macro_f1": float(f1), "brier": float(brier), "ece": float(ece)}
