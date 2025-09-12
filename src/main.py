"""src/main.py
Main orchestration entry-point – updated to comply with mandatory directory
layout (.research/iteration2) and image path requirement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

# Third-party modules used only in full experiment mode.  Kept here to fail fast
# on missing deps.
import requests
import torch  # noqa: F401 – imported solely to guarantee PyTorch availability

from .preprocess import run_preprocessing_pipeline
from .train import run_training_pipeline
from .evaluate import run_evaluation_pipeline

# -----------------------------------------------------------------------------
# Paths & constants (iteration **2** as mandated by instructions)
# -----------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _PROJECT_ROOT / "config"
_RESULT_DIR = _PROJECT_ROOT / ".research" / "iteration2"
_IMAGE_DIR = _RESULT_DIR / "images"

_SMOKE_CONFIG = _CONFIG_DIR / "smoke_test.yaml"
_FULL_CONFIG = _CONFIG_DIR / "full_experiment.yaml"
_DATA_DIR = _PROJECT_ROOT / "data"

_RESULT_DIR.mkdir(parents=True, exist_ok=True)
_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
_DATA_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Utility helpers (unchanged apart from path update comments)
# -----------------------------------------------------------------------------

def sha256sum(file_path: Path) -> str:
    """Compute SHA-256 for a local file (chunked reading)."""
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def human_error(msg: str) -> None:  # pragma: no cover – convenience helper
    print("\n*** FATAL: " + msg + "\n", file=sys.stderr)
    sys.exit(1)

# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------

def _load_cfg(path: Path) -> Dict[str, Any]:
    if not path.exists():
        human_error(f"Configuration file not found at {path} – aborting.")
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:  # Broad except ok for top-level config parse
        human_error(f"Failed to parse YAML at {path}: {e}")


def _validate_datasets(cfg: Dict[str, Any]) -> None:
    missing = [name for name, info in cfg.get("datasets", {}).items()
               if not info.get("url") or str(info.get("url")).strip().lower() in {"", "null"}]
    if missing:
        human_error("Dataset URLs are missing for: " + ", ".join(missing))


def _validate_models(cfg: Dict[str, Any]) -> None:
    models = cfg.get("models", {})
    if not models:
        human_error("No models specified in configuration.")
    for name, info in models.items():
        if "hf_hub_id" not in info:
            human_error(f"Model entry '{name}' lacks 'hf_hub_id'.")

# -----------------------------------------------------------------------------
# Pipelines
# -----------------------------------------------------------------------------

def _run_smoke_test(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a quick end-to-end pass without external network usage."""
    _validate_models(cfg)
    _validate_datasets(cfg)

    preprocess_metrics = run_preprocessing_pipeline(cfg)
    train_metrics = run_training_pipeline(cfg)
    eval_metrics = run_evaluation_pipeline(cfg)

    result = {
        "phase": "smoke_test",
        "preprocess": preprocess_metrics,
        "train": train_metrics,
        "eval": eval_metrics,
        "status": "passed",
    }
    return result


def _run_full_experiment(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Replicates the strict behaviour from the original experiment script."""
    _validate_models(cfg)
    _validate_datasets(cfg)

    # Reachability check for first dataset.
    first_ds_name, first_ds = next(iter(cfg["datasets"].items()))
    first_url = first_ds["url"]
    print(f"Checking reachability of dataset '{first_ds_name}' at {first_url} …", flush=True)
    try:
        head_resp = requests.head(first_url, timeout=10, allow_redirects=True)
    except Exception as e:
        human_error(f"Failed to contact {first_url}: {e}")
    if head_resp.status_code >= 400:
        human_error(f"Dataset URL {first_url} returned HTTP {head_resp.status_code}.")

    preprocess_metrics = run_preprocessing_pipeline(cfg)
    train_metrics = run_training_pipeline(cfg)
    eval_metrics = run_evaluation_pipeline(cfg)

    result = {
        "phase": "full_experiment",
        "preprocess": preprocess_metrics,
        "train": train_metrics,
        "eval": eval_metrics,
        "status": "completed (logic stub – integrate real models)",
    }
    return result

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _dump_results(result: Dict[str, Any], tag: str) -> None:
    out_path = _RESULT_DIR / f"{tag}_results.json"
    try:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        human_error(f"Unable to write results to {out_path}: {e}")
    print("\n=== Result JSON (" + tag + ") ===")
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
        result = _run_smoke_test(cfg)
        _dump_results(result, "smoke_test")
    else:
        result = _run_full_experiment(cfg)
        _dump_results(result, "full_experiment")


if __name__ == "__main__":
    main()
