# src/main.py
"""Entry-point for running TIGER-Lite experiments.

Usage
-----
Smoke test only
    uv run python -m src.main --smoke-test

Full experiment only
    uv run python -m src.main --full-experiment
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any, Dict

import torch
import yaml

from .preprocess import build_tasks
from .train import continual_train, ResourceViolation
from .models import TigerLite

# -----------------------------------------------------------------------------
#   Utilities
# -----------------------------------------------------------------------------

_CONFIG_DIR = pathlib.Path("config")


def _load_config(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file '{path}' not found.")
    return yaml.safe_load(path.read_text())


# -----------------------------------------------------------------------------
#   Argument parsing
# -----------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser("TIGER-Lite experiment runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run quick smoke test")
    group.add_argument("--full-experiment", action="store_true", help="Run full experiment")
    return parser.parse_args()


# -----------------------------------------------------------------------------
#   Main orchestration
# -----------------------------------------------------------------------------

def _run(cfg_name: str):
    cfg_path = _CONFIG_DIR / cfg_name
    cfg = _load_config(cfg_path)["experiment"]

    print("================  EXPERIMENT DESCRIPTION  ================")
    print(json.dumps(cfg, indent=2))
    print("==========================================================")

    save_root = pathlib.Path(cfg["save_dir"])

    for seed in cfg["seeds"]:
        torch.manual_seed(seed)
        for ds_name in cfg["datasets"]:
            tasks = build_tasks(ds_name, max_tasks=cfg.get("max_tasks"))
            model = TigerLite()

            # budgets may be a list (multiple settings) or a mapping.
            budgets = cfg["budgets"][0] if isinstance(cfg["budgets"], list) else cfg["budgets"]

            try:
                json_path, figs = continual_train(
                    model,
                    tasks,
                    {**budgets, **cfg},
                    save_root / f"seed{seed}_{ds_name}",
                )
            except ResourceViolation as err:
                print(f"EXPERIMENT FAILED – resource violation: {err}", file=sys.stderr)
                sys.exit(1)

            # ---------------  STDOUT: required JSON content ---------------
            print("\n---  Numerical Results  ---")
            print(pathlib.Path(json_path).read_text())
            print("---  Figures  ---")
            for fig in figs:
                print(fig)


# -----------------------------------------------------------------------------
#   Entry point
# -----------------------------------------------------------------------------


def main():
    args = _parse_args()
    cfg_file = "smoke_test.yaml" if args.smoke_test else "full_experiment.yaml"
    _run(cfg_file)


if __name__ == "__main__":
    main()
