# src/main.py
"""CLI entry-point for SAFE-FUSE-Ψ reference implementation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import sys

# Ensure the project source directory is importable when the script is executed
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch  # noqa: E402  pylint: disable=wrong-import-position

from train import load_cfg, LocalTrainer  # noqa: E402  pylint: disable=wrong-import-position
from preprocess import load_graphs  # noqa: E402  pylint: disable=wrong-import-position
from evaluate import log_metrics  # noqa: E402  pylint: disable=wrong-import-position

# ---------------------------------------------------------------------------
# Mandatory research output paths
# ---------------------------------------------------------------------------
JSON_DIR = Path(".research/iteration3").resolve()
JSON_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper to run a single silo (non-federated for smoke/full demo)
# ---------------------------------------------------------------------------

def _run(cfg_path: Path):
    cfg = load_cfg(cfg_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for ds_name, ds_cfg in cfg.datasets.items():
        graphs = load_graphs(ds_cfg)

        for eps in cfg.dp.epsilons:
            trainer = LocalTrainer(
                silo_id=f"{ds_name}_s0",
                graphs=graphs,
                cfg=cfg,
                epsilon=eps,
                delta=cfg.dp.delta,
                device=device,
            )

            for epoch in range(cfg.training.epochs_per_snapshot):
                print(f"[Dataset {ds_name}] ε={eps} – epoch {epoch+1}/{cfg.training.epochs_per_snapshot}")
                trainer.train_one_epoch()

            metrics = trainer.evaluate()
            log_metrics(ds_name, metrics)

            # ----------------------------------------------------------------
            # Save JSON to mandatory directory
            # ----------------------------------------------------------------
            out = {
                "dataset": ds_name,
                "epsilon": eps,
                "metrics": metrics,
            }
            json_path = JSON_DIR / f"{cfg.name}_{ds_name}_eps{eps}.json"
            with json_path.open("w", encoding="utf8") as fh:
                json.dump(out, fh, indent=2)

            # Print to stdout for verification (as requested)
            print(json.dumps(out, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("safe-fuse-psi")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run the quick smoke test configuration.")
    group.add_argument("--full-experiment", action="store_true", help="Run the full experiment configuration.")
    group.add_argument("--config", type=str, help="Path to a custom YAML experiment configuration.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke_test:
        _run(Path("config/smoke_test.yaml"))
    elif args.full_experiment:
        _run(Path("config/full_experiment.yaml"))
    else:
        _run(Path(args.config))
