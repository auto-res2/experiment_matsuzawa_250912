"""src/main.py
Command-line entry point that now executes a *complete* experiment pipeline:
1. Synthetic data generation (src.preprocess.load_and_preprocess_data)
2. Training (src.train.train_model)
3. Evaluation (src.evaluate.evaluate_model)
4. JSON result persistence under .research/iteration4/

This fulfils the requirement that a numerical artefact is produced for both
the smoke-test and full-experiment flags.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from .evaluate import evaluate_model
from .preprocess import load_and_preprocess_data
from .train import train_model

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
    """Load YAML configuration for *mode* ("smoke_test" | "full_experiment")."""
    config_dir = Path(__file__).resolve().parent.parent / "config"
    filename = "smoke_test.yaml" if mode == "smoke_test" else "full_experiment.yaml"
    cfg_path = config_dir / filename
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _save_results(results: Dict[str, Any], label: str) -> None:
    """Persist *results* to .research/iteration4/<label>.json and echo them."""
    results_dir = Path(__file__).resolve().parent.parent / ".research" / "iteration4"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{label}.json"
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    # Console echo for CI logs
    logger.info("%s", json.dumps(results, indent=2))


# -----------------------------------------------------------------------------
# Main experiment orchestration
# -----------------------------------------------------------------------------

def _run_pipeline(mode: str) -> None:  # noqa: D401
    """End-to-end execution for the given *mode*."""
    config = _load_config(mode)

    # -------------------- data --------------------
    train_loader, val_loader = load_and_preprocess_data(config)

    # ------------------- train --------------------
    model, train_metrics = train_model(config=config, train_loader=train_loader, val_loader=val_loader)

    # -------------- optional evaluation -----------
    eval_metrics = evaluate_model(model=model, val_loader=val_loader, config=config)

    # -------------- merge & persist ---------------
    merged = {**train_metrics, **eval_metrics}
    _save_results(merged, label=mode)


def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Synthetic experiment runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run quick smoke-test (1 epoch, 100 samples)")
    group.add_argument("--full-experiment", action="store_true", help="Run full synthetic experiment")
    args = parser.parse_args()

    mode = "smoke_test" if args.smoke_test else "full_experiment"
    logger.info("=== [%s] Experiment start ===", mode)
    _run_pipeline(mode)
    logger.info("=== [%s] Experiment end ===", mode)


if __name__ == "__main__":
    main()
