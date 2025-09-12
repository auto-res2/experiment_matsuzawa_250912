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

from .evaluate import init_logger
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
# Orchestration logic
# --------------------------------------------------------------------------------

def _execute(cfg: Dict[str, Any]):
    """Execute all experiments sequentially with the given configuration."""
    out_dir = Path(cfg["experiment"]["output_dir"]) / cfg["experiment"]["name"]
    init_logger(out_dir / "metrics.json")

    data_root = Path("data")
    data_root.mkdir(parents=True, exist_ok=True)

    for ExpCls in (Align16Experiment, HIL12Experiment, Face80Experiment):
        try:
            ExpCls(cfg, data_root).run()
        except RuntimeError as e:
            print(f"\n[ABORT] {ExpCls.name}: {e}\n")
            sys.exit(1)


# --------------------------------------------------------------------------------
# Entry-point
# --------------------------------------------------------------------------------

def main():
    args = _parse_args()

    root_cfg_dir = Path(__file__).resolve().parent.parent / "config"

    if args.smoke_test:
        cfg_path = root_cfg_dir / "smoke_test.yaml"
        cfg = _load_cfg(cfg_path)
        # Mark config so that experiments can enter fast path
        cfg["smoke_test"] = True
        _execute(cfg)

    elif args.full_experiment:
        # Phase 1 – smoke test ------------------------------------------------
        smoke_cfg_path = root_cfg_dir / "smoke_test.yaml"
        smoke_cfg = _load_cfg(smoke_cfg_path)
        smoke_cfg["smoke_test"] = True
        print("\n================  Smoke Test  ================\n")
        _execute(smoke_cfg)

        # Phase 2 – full experiment -----------------------------------------
        full_cfg_path = root_cfg_dir / "full_experiment.yaml"
        full_cfg = _load_cfg(full_cfg_path)
        full_cfg["smoke_test"] = False
        print("\n==============  Full Experiment  =============\n")
        _execute(full_cfg)


if __name__ == "__main__":
    main()
