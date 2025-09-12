"""
main.py – command-line interface for ReFuse-CL experiments
──────────────────────────────────────────────────────────
Supports the required execution patterns:
    uv run python -m src.main --smoke-test
    uv run python -m src.main --full-experiment
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

from .train import run_experiment

logger = logging.getLogger("ReFuse-CL")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

ROOT = Path(__file__).resolve().parent.parent  # project root (…/src/..)
CONFIG_DIR = ROOT / "config"
RESEARCH_DIR = ROOT / ".research" / "iteration1"
IMAGES_DIR = RESEARCH_DIR / "images"


# -----------------------------------------------------------------------------
#  Helper: load YAML configuration
# -----------------------------------------------------------------------------

def _load_cfg(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file {path} not found (cwd={Path.cwd()}). "
            "Make sure the repository has been initialised correctly."
        )
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# -----------------------------------------------------------------------------
#  CLI entry-point
# -----------------------------------------------------------------------------

def main() -> None:  # noqa: C901 – keep single entry-point for clarity
    parser = argparse.ArgumentParser(description="Run ReFuse-CL experiment suite")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--smoke-test", action="store_true", help="run quick validation run")
    grp.add_argument("--full-experiment", action="store_true", help="run full benchmark")
    args = parser.parse_args()

    cfg_file = "smoke_test.yaml" if args.smoke_test else "full_experiment.yaml"
    cfg_path = CONFIG_DIR / cfg_file
    cfg = _load_cfg(cfg_path)

    # ------------------------------------------------------------------
    #  Prepare output directories
    # ------------------------------------------------------------------
    for p in [RESEARCH_DIR, IMAGES_DIR]:
        p.mkdir(parents=True, exist_ok=True)

    # Persist config snapshot for reproducibility
    snapshot_name = (
        cfg_path.stem + "_snapshot_" + datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + ".yaml"
    )
    shutil.copy(cfg_path, RESEARCH_DIR / snapshot_name)

    # ------------------------------------------------------------------
    #  Execute experiment
    # ------------------------------------------------------------------
    try:
        run_experiment(cfg)
        status = "completed"
    except Exception as err:  # noqa: BLE001 – broad except keeps CLI alive
        logger.error("Experiment failed: %s", err)
        status = f"failed: {err}"

    # ------------------------------------------------------------------
    #  Minimal result JSON for bookkeeping / smoke validation
    # ------------------------------------------------------------------
    experiment_name = cfg.get("experiment_name") or cfg_path.stem
    result_obj = {
        "experiment_name": experiment_name,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "status": status,
    }

    json_path = RESEARCH_DIR / f"{cfg_path.stem}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    try:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(result_obj, fh, indent=2)
    except OSError as err:  # pragma: no cover – disk full / perms
        logger.warning("Could not write result JSON %s: %s", json_path, err)

    # Always print to stdout for CI visibility
    print(json.dumps(result_obj, indent=2))


if __name__ == "__main__":
    main()
