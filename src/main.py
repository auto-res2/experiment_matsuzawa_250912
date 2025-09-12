"""src/main.py – command-line entry point

This script supports the following invocations:

    # Smoke test only
    uv run python -m src.main --smoke-test

    # Full experiment only
    uv run python -m src.main --full-experiment

If *no* flag is provided the script will automatically execute the smoke test
first and, provided it completes without unhandled exceptions, launch the full
experiment afterwards.
"""
from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace
from typing import List

import random
import traceback
from pathlib import Path

import yaml

# Local relative imports (package "src")
from .evaluate import ExperimentRunner

# -----------------------------------------------------------------------------
# Helper – YAML → SimpleNamespace (avoids additional file just for dataclasses)
# -----------------------------------------------------------------------------

def _cfg_from_yaml(path: Path) -> List[SimpleNamespace]:
    with path.open("r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp)

    experiments_cfg: List[SimpleNamespace] = []
    for exp in raw.get("experiments", []):
        # Flatten dictionary into attribute namespace for nicer dot-access later
        experiments_cfg.append(SimpleNamespace(**exp))
    random_seed = raw.get("random_seed", 23)
    random.seed(random_seed)
    return experiments_cfg


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------

def main() -> None:  # noqa: D401  (imperative CLI)
    root_dir = Path(__file__).resolve().parents[1]
    cfg_dir = root_dir / "config"

    smoke_cfg = cfg_dir / "smoke_test.yaml"
    full_cfg = cfg_dir / "full_experiment.yaml"

    if not smoke_cfg.exists() or not full_cfg.exists():
        raise FileNotFoundError(
            "Could not locate configuration YAMLs. Ensure that 'config/' contains "
            "'smoke_test.yaml' and 'full_experiment.yaml'."
        )

    # ------------------------ CLI parsing -------------------------------
    parser = argparse.ArgumentParser(description="CADENCE experimental launcher")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--smoke-test", action="store_true", help="run the smoke test only")
    group.add_argument("--full-experiment", action="store_true", help="run the full experiment only")
    args = parser.parse_args()

    try:
        if args.smoke_test:  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            for exp in _cfg_from_yaml(smoke_cfg):
                ExperimentRunner(exp).run()

        elif args.full_experiment:  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            for exp in _cfg_from_yaml(full_cfg):
                ExperimentRunner(exp).run()

        else:  # Default behaviour – two-phase execution
            print("[main] No flag provided – running smoke test first …")
            for exp in _cfg_from_yaml(smoke_cfg):
                ExperimentRunner(exp).run()
            print("[main] Smoke test finished successfully – commencing full experiment …")
            for exp in _cfg_from_yaml(full_cfg):
                ExperimentRunner(exp).run()

    except Exception as exc:  # pylint: disable=broad-except
        # We *never* swallow exceptions silently – print traceback + exit 1
        traceback.print_exc()
        print(f"[main] Fatal error: {exc}")
        sys.exit(1)


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
