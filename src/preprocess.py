"""src/preprocess.py
Generate a synthetic 2-D classification dataset (two Gaussian blobs) so that
the training loop has something to chew on during CI.  The function returns
PyTorch ``DataLoader`` objects for train & validation splits.  The dataset size
is controlled via the YAML config (``data.subset``).  This lightweight setup
avoids external dataset downloads and therefore keeps the test environment
self-contained.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def _make_blob(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create *n* samples from two separable 2-D Gaussians."""
    n_per_class = n // 2
    cov = [[0.1, 0.0], [0.0, 0.1]]
    x0 = np.random.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=n_per_class)
    x1 = np.random.multivariate_normal(mean=[1.0, 1.0], cov=cov, size=n_per_class)
    X = np.vstack([x0, x1]).astype(np.float32)
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)]).astype(np.int64)
    return X, y


def load_and_preprocess_data(  # noqa: D401
    config: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader]:
    """Create DataLoaders for a synthetic binary-classification task."""

    subset = config.get("data", {}).get("subset")
    subset = int(subset) if subset is not None else 1000  # default size

    logger.info("Generating synthetic dataset with %d samples…", subset)
    X, y = _make_blob(subset)

    # Split 80/20
    split = int(0.8 * subset)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    batch_size = int(config.get("train", {}).get("batch_size", 32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
