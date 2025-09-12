"""
main.py
~~~~~~~
The single public entry-point.  Supports either a quick smoke test or the full
experimental suite and is executable with

    uv run python -m src.main --smoke-test       # quick CI check
    uv run python -m src.main --full-experiment  # full research run (includes smoke)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime

import yaml

from .evaluate import Experiment1, Experiment2, Experiment3

# ---------------------------------------------------------------------------
#  Command-line argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="STARLING-Duo experimental driver")
flag = parser.add_mutually_exclusive_group(required=True)
flag.add_argument("--smoke-test", action="store_true", help="run only the tiny CI smoke test")
flag.add_argument("--full-experiment", action="store_true", help="run smoke test first and then the full suite")
args = parser.parse_args()

# ---------------------------------------------------------------------------
#  Path bootstrap  (config  /  research-output)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"
SMOKE_CFG_PATH = CONFIG_DIR / "smoke_test.yaml"
FULL_CFG_PATH = CONFIG_DIR / "full_experiment.yaml"

if not SMOKE_CFG_PATH.exists() or not FULL_CFG_PATH.exists():
    sys.exit("Configuration files are missing – run `git checkout` again or recreate them.")

# ---------------------------------------------------------------------------
#  Helper that executes a single config file
# ---------------------------------------------------------------------------
runner_lookup = {1: Experiment1, 2: Experiment2, 3: Experiment3}

def _run_config(path: Path):
    cfg = yaml.safe_load(path.read_text())
    for exp_id in cfg["experiments"]:
        runner_cls = runner_lookup.get(exp_id)
        if runner_cls is None:
            raise RuntimeError(f"Unknown experiment id {exp_id} in {path}")
        runner_cls(cfg).run()


# ---------------------------------------------------------------------------
#  Orchestration logic (two-phase if --full-experiment)
# ---------------------------------------------------------------------------
if args.smoke_test:
    _run_config(SMOKE_CFG_PATH)
elif args.full_experiment:
    print("Running preliminary smoke test …")
    _run_config(SMOKE_CFG_PATH)
    print("Smoke test passed – launching full experiment\n")
    _run_config(FULL_CFG_PATH)

print("\nAll experiments finished at", datetime.utcnow().isoformat(" ", "seconds"))
