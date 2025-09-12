"""src/main.py
Command-line entry point.  Supports two modes:
  • Smoke test  – `python -m src.main --smoke-test`
  • Full run    – `python -m src.main --full-experiment`

The *full* run first launches a quick smoke pass; if that succeeds we proceed
with the heavy experiment.  All config files live under *config/*.  If they are
missing we create default templates on the fly so users can edit them later.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Dict, Any

import yaml

from .evaluate import experiment1

# -----------------------------------------------------------------------------
# Directories & default configuration templates
# -----------------------------------------------------------------------------
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

SMOKE_TEMPLATE: Dict[str, Any] = {
    "experiment": "smoke",
    "dataset_split": "test[:1%]",
    "dataset_proportion": 0.01,
    "teacher_model": "Helsinki-NLP/opus-mt-en-de",
    "quadron_model": "quadron-ai/quadron-dm-text-wmt14",
}

FULL_TEMPLATE: Dict[str, Any] = {
    "experiment": "full",
    "dataset_split": "test[:100%]",
    "dataset_proportion": 1.0,
    "teacher_model": "Helsinki-NLP/opus-mt-en-de",
    "quadron_model": "quadron-ai/quadron-dm-text-wmt14",
}


# -----------------------------------------------------------------------------
# Tiny helper to (re)generate missing YAMLs so users can tweak them later.
# -----------------------------------------------------------------------------

def _write_yaml_if_missing(path: pathlib.Path, content: Dict[str, Any]):
    if not path.exists():
        with path.open("w", encoding="utf-8") as fp:
            yaml.dump(content, fp, sort_keys=False)


# -----------------------------------------------------------------------------
# Main orchestration logic
# -----------------------------------------------------------------------------

def _load_cfg(name: str) -> Dict[str, Any]:
    cfg_path = CONFIG_DIR / name
    if not cfg_path.exists():
        sys.exit(f"Config file '{cfg_path}' does not exist – aborting.")
    with cfg_path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _run_with_cfg(cfg_name: str):
    cfg = _load_cfg(cfg_name)
    experiment1(cfg)


def main() -> None:  # noqa: D401 – simple launcher
    # Ensure default config stubs exist
    _write_yaml_if_missing(CONFIG_DIR / "smoke_test.yaml", SMOKE_TEMPLATE)
    _write_yaml_if_missing(CONFIG_DIR / "full_experiment.yaml", FULL_TEMPLATE)

    parser = argparse.ArgumentParser(description="QuADRoN-DM experimental runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="quick validation pass")
    group.add_argument("--full-experiment", action="store_true", help="run smoke test then full experiment")
    args = parser.parse_args()

    if args.smoke_test:
        print("[INFO] Launching smoke test…")
        _run_with_cfg("smoke_test.yaml")
        return

    # Full experiment requested – run smoke test first for safety
    print("[INFO] Running mandatory smoke test before full experiment…")
    try:
        _run_with_cfg("smoke_test.yaml")
    except SystemExit as exc:  # propagate failure code but with message
        print("[ERROR] Smoke test failed → skipping full experiment.")
        raise

    print("[INFO] Smoke test succeeded – starting full experiment…")
    _run_with_cfg("full_experiment.yaml")


if __name__ == "__main__":  # pragma: no cover
    main()
