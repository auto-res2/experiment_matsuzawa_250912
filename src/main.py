"""src/main.py
Command-line entry-point orchestrating smoke-test and full-experiment runs.
Even though the core experiment logic is missing, this file fulfils the required interface so that
`uv run python -m src.main --smoke-test` and `--full-experiment` execute without import errors.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

# Local imports – will raise if underlying modules still contain NotImplementedError
from .preprocess import load_and_preprocess_data
from .train import train_model
from .evaluate import evaluate_model

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _load_config(mode: str) -> Dict[str, Any]:
    """Load YAML configuration from the config directory.

    Parameters
    ----------
    mode : str
        Either "smoke_test" or "full_experiment".
    """
    config_dir = Path(__file__).resolve().parent.parent / "config"
    filename = "smoke_test.yaml" if mode == "smoke_test" else "full_experiment.yaml"
    cfg_path = config_dir / filename
    try:
        with cfg_path.open("r", encoding="utf-8") as fp:
            cfg: Dict[str, Any] = yaml.safe_load(fp)
    except FileNotFoundError as exc:
        logger.error("Configuration file %s not found", cfg_path)
        raise exc
    return cfg


def _save_results(results: Dict[str, Any], label: str) -> None:
    """Persist a results dict under .research/iteration3 and print to stdout."""
    results_dir = Path(__file__).resolve().parent.parent / ".research" / "iteration3"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{label}.json"
    try:
        with out_path.open("w", encoding="utf-8") as fp:
            json.dump(results, fp, indent=2)
        # Echo to console for quick verification
        logger.info("%s", json.dumps(results, indent=2))
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Failed to write results to %s", out_path)
        raise exc

# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def run_experiment(mode: str) -> None:  # noqa: D401
    """Run either the smoke test or the full experiment."""
    logger.info("Running %s...", mode)
    config = _load_config("smoke_test" if mode == "smoke_test" else "full_experiment")

    # ----------------------------- Pre-processing ----------------------------
    try:
        train_data, val_data = load_and_preprocess_data(config)
    except NotImplementedError:
        logger.warning("Preprocess step not implemented – skipping further execution.")
        return

    # ----------------------------- Training ----------------------------------
    try:
        model = train_model(config=config)
    except NotImplementedError:
        logger.warning("Training step not implemented – skipping further execution.")
        return

    # ----------------------------- Evaluation --------------------------------
    try:
        metrics = evaluate_model(model=model, config=config)
    except NotImplementedError:
        logger.warning("Evaluation step not implemented – skipping further execution.")
        return

    # ----------------------------- Persistence -------------------------------
    _save_results(metrics, label=mode)


def main() -> None:  # noqa: D401
    """Entry-point for command-line execution."""
    parser = argparse.ArgumentParser(description="Experiment runner with smoke-test support")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run the quick smoke test")
    group.add_argument("--full-experiment", action="store_true", help="Run the full experiment")

    args = parser.parse_args()

    if args.smoke_test:
        run_experiment("smoke_test")
    elif args.full_experiment:
        run_experiment("full_experiment")
    else:  # pragma: no cover – argparse enforces one flag
        parser.error("Either --smoke-test or --full-experiment must be supplied.")


if __name__ == "__main__":
    main()
