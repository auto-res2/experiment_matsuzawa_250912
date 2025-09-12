"""
main.py – Single entry-point.  Supports the following command patterns:

    uv run python -m src.main --smoke-test
    uv run python -m src.main --full-experiment
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import yaml

from .evaluate import init_logger, close_logger
from .train import Align16Experiment, HIL12Experiment, Face80Experiment

# --------------------------------------------------------------------------------
# Helper for loading YAML configurations
# --------------------------------------------------------------------------------

def _load_cfg(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise RuntimeError(f"Configuration file '{path}' not found.") from e
    return cfg


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIPHER-Ω experimental runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run quick smoke test only")
    group.add_argument("--full-experiment", action="store_true", help="Run smoke test and full experiment")
    return parser.parse_args()


# --------------------------------------------------------------------------------
# Orchestration helpers
# --------------------------------------------------------------------------------

_EXPERIMENT_CLASSES = (
    Align16Experiment,
    HIL12Experiment,
    Face80Experiment,
)


def _run_experiments(cfg: Dict[str, Any]):
    """Run each experiment with its *own* logger/JSON file."""
    root_out = Path(cfg["experiment"]["output_dir"]) / cfg["experiment"]["name"]
    data_root = Path("data")
    data_root.mkdir(parents=True, exist_ok=True)

    for ExpCls in _EXPERIMENT_CLASSES:
        # Ensure a fresh logger instance per experiment -------------------
        close_logger()
        exp_out_file = root_out / f"{ExpCls.name}.json"
        init_logger(exp_out_file)

        try:
            ExpCls(cfg, data_root).run()
        except RuntimeError as e:
            # Flush current logger before aborting so that partial results are not lost
            close_logger()
            print(f"\n[ABORT] {ExpCls.name}: {e}\n")
            sys.exit(1)

        # Graceful shutdown of logger to persist results before next experiment
        close_logger()


# --------------------------------------------------------------------------------
# Entry-point
# --------------------------------------------------------------------------------

def main():
    args = _parse_args()

    root_cfg_dir = Path(__file__).resolve().parent.parent / "config"

    if args.smoke_test:
        cfg_path = root_cfg_dir / "smoke_test.yaml"
        cfg = _load_cfg(cfg_path)
        cfg["smoke_test"] = True  # experiments read this flag
        _run_experiments(cfg)

    elif args.full_experiment:
        # ----------------------------------------------------------------- Phase 1 (smoke)
        smoke_cfg_path = root_cfg_dir / "smoke_test.yaml"
        smoke_cfg = _load_cfg(smoke_cfg_path)
        smoke_cfg["smoke_test"] = True
        print("\n================  Smoke Test  ================\n")
        _run_experiments(smoke_cfg)

        # ----------------------------------------------------------------- Phase 2 (full)
        full_cfg_path = root_cfg_dir / "full_experiment.yaml"
        full_cfg = _load_cfg(full_cfg_path)
        full_cfg["smoke_test"] = False
        print("\n==============  Full Experiment  =============\n")
        _run_experiments(full_cfg)


if __name__ == "__main__":
    main()
