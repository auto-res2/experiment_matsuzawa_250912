"""src/main.py
Entry-point script for running COSMOS-Diff experiments via CLI.

Usage:
  python -m src.main --smoke-test        # quick CI validation
  python -m src.main --full-experiment   # full run (longer)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import yaml

# -----------------------------------------------------------------------------
#  Robust import of the training sub-module.
#  Using an absolute import avoids the static-analysis error raised for
#  relative imports in a top-level script.  When the script is executed via
#  `python src/main.py`, the parent directory is not on `sys.path`; we append it
#  on-the-fly so that `import src` succeeds.  When executed with
#  `python -m src.main`, the path is already correct and the import just works.
# -----------------------------------------------------------------------------
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from src import train  # noqa: E402 – import after path fix

ROOT = Path(__file__).resolve().parent.parent
CFG_DIR = ROOT / "config"

# -----------------------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------------------

def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        sys.exit(f"[FATAL] Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -----------------------------------------------------------------------------
#  CLI
# -----------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="COSMOS-Diff experiment launcher")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke-test", action="store_true", help="Run quick smoke test")
    g.add_argument("--full-experiment", action="store_true", help="Run full experiment")
    return p.parse_args(argv)


# -----------------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    cfg_file = (
        CFG_DIR / "smoke_test.yaml" if args.smoke_test else CFG_DIR / "full_experiment.yaml"
    )
    cfg = _load_yaml(cfg_file)

    print(f"Loaded configuration: {cfg_file}\n")

    # ------------------------------------------------------------------
    # Experiment executions
    # ------------------------------------------------------------------
    train.run_experiment_1(cfg)
    train.run_experiment_2(cfg)
    train.run_experiment_3(cfg)


if __name__ == "__main__":
    main()
