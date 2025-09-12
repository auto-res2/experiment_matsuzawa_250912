"""src/main.py
Main orchestration entry-point.  Updated to comply with *iteration3* directory
requirements and to reflect the new, non-trivial metric pipelines.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

# Third-party modules used only in full-experiment mode.  Kept to fail fast on
# missing dependencies.
import requests  # noqa: F401
import torch  # noqa: F401 – ensure PyTorch is importable

from .preprocess import run_preprocessing_pipeline
from .train import run_training_pipeline
from .evaluate import run_evaluation_pipeline

# ----------------------------------------------------------------------------
# Paths & constants (iteration **3**)                                          |
# ----------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _PROJECT_ROOT / "config"
_RESULT_DIR = _PROJECT_ROOT / ".research" / "iteration3"
_IMAGE_DIR = _RESULT_DIR / "images"

_SMOKE_CONFIG = _CONFIG_DIR / "smoke_test.yaml"
_FULL_CONFIG = _CONFIG_DIR / "full_experiment.yaml"
_DATA_DIR = _PROJECT_ROOT / "data"

_RESULT_DIR.mkdir(parents=True, exist_ok=True)
_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
_DATA_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# Utility helpers                                                              |
# ----------------------------------------------------------------------------

def _sha256sum(file_path: Path) -> str:  # pragma: no cover – helper
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _human_error(msg: str) -> None:  # pragma: no cover – convenience helper
    print(f"\n*** FATAL: {msg}\n", file=sys.stderr, flush=True)
    sys.exit(1)

# ----------------------------------------------------------------------------
# Configuration helpers                                                        |
# ----------------------------------------------------------------------------

def _load_cfg(path: Path) -> Dict[str, Any]:
    if not path.exists():
        _human_error(f"Configuration file not found at {path} – aborting.")
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:  # Broad except acceptable for top-level config parse
        _human_error(f"Failed to parse YAML at {path}: {e}")


def _validate_datasets(cfg: Dict[str, Any]) -> None:
    missing = [name for name, info in cfg.get("datasets", {}).items() if not info.get("url")]
    if missing:
        _human_error("Dataset URLs are missing for: " + ", ".join(missing))


def _validate_models(cfg: Dict[str, Any]) -> None:
    models = cfg.get("models", {})
    if not models:
        _human_error("No models specified in configuration.")
    for name, info in models.items():
        if "hf_hub_id" not in info:
            _human_error(f"Model entry '{name}' lacks 'hf_hub_id'.")

# ----------------------------------------------------------------------------
# Pipelines                                                                    |
# ----------------------------------------------------------------------------

def _run_smoke_test(cfg: Dict[str, Any]) -> Dict[str, Any]:
    _validate_models(cfg)
    _validate_datasets(cfg)

    preprocess_metrics = run_preprocessing_pipeline(cfg)
    train_metrics = run_training_pipeline(cfg)
    eval_metrics = run_evaluation_pipeline(cfg)

    return {
        "phase": "smoke_test",
        "preprocess": preprocess_metrics,
        "train": train_metrics,
        "eval": eval_metrics,
        "status": "passed",
    }


def _run_full_experiment(cfg: Dict[str, Any]) -> Dict[str, Any]:
    _validate_models(cfg)
    _validate_datasets(cfg)

    # Reachability check for the first dataset to fail fast on network issues.
    first_name, first_ds = next(iter(cfg["datasets"].items()))
    first_url = first_ds["url"]
    print(f"Checking reachability of dataset '{first_name}' at {first_url} …", flush=True)
    try:
        head = requests.head(first_url, timeout=10, allow_redirects=True)
        if head.status_code >= 400:
            _human_error(f"Dataset URL {first_url} returned HTTP {head.status_code}.")
    except Exception as e:
        _human_error(f"Failed to contact {first_url}: {e}")

    preprocess_metrics = run_preprocessing_pipeline(cfg)
    train_metrics = run_training_pipeline(cfg)
    eval_metrics = run_evaluation_pipeline(cfg)

    return {
        "phase": "full_experiment",
        "preprocess": preprocess_metrics,
        "train": train_metrics,
        "eval": eval_metrics,
        "status": "completed",
    }

# ----------------------------------------------------------------------------
# CLI                                                                          |
# ----------------------------------------------------------------------------

def _dump_results(result: Dict[str, Any], tag: str) -> None:
    out_path = _RESULT_DIR / f"{tag}_results.json"
    try:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        _human_error(f"Unable to write results to {out_path}: {e}")

    print(f"\n=== Result JSON ({tag}) ===")
    print(json.dumps(result, indent=2))


def main() -> None:  # pragma: no cover – top-level script
    parser = argparse.ArgumentParser(description="CIPHER-Ω experiment runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run quick validation only")
    group.add_argument("--full-experiment", action="store_true", help="Run full experiment")

    args = parser.parse_args()

    cfg_path = _SMOKE_CONFIG if args.smoke_test else _FULL_CONFIG
    cfg = _load_cfg(cfg_path)

    if args.smoke_test:
        res = _run_smoke_test(cfg)
        _dump_results(res, "smoke_test")
    else:
        res = _run_full_experiment(cfg)
        _dump_results(res, "full_experiment")


if __name__ == "__main__":
    main()
