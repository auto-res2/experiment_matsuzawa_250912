"""
main.py – command-line orchestrator
Supports
  uv run python -m src.main --smoke-test
  uv run python -m src.main --full-experiment
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

# ---------------------------------------------------------------------------
#  Locate repository root and make sure `src/` is importable as namespace pkg
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_ROOT: Path = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# ---------------------------------------------------------------------------
#  CLI parsing – mutually-exclusive run-modes
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="COSMOS-Diff experimental runner")
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument("--smoke-test", action="store_true", help="run quick CI sanity-check")
mode.add_argument("--full-experiment", action="store_true", help="run the full experimental suite")
args = parser.parse_args()

# ---------------------------------------------------------------------------
#  Configuration loading
# ---------------------------------------------------------------------------
if args.smoke_test:
    cfg_path = REPO_ROOT / "config" / "smoke_test.yaml"
else:
    cfg_path = REPO_ROOT / "config" / "full_experiment.yaml"

if not cfg_path.exists():
    raise FileNotFoundError(f"Config file not found: {cfg_path}")

CONFIG: Dict[str, Any] = yaml.safe_load(cfg_path.read_text())

# ---------------------------------------------------------------------------
#  Directory preparation (results / images live under .research/iteration6/…)
# ---------------------------------------------------------------------------
RESEARCH_DIR: Path = REPO_ROOT / ".research" / "iteration6"
IMAGES_DIR: Path = RESEARCH_DIR / "images"
RESULTS_DIR: Path = RESEARCH_DIR
for d in (IMAGES_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
#  Environment & dependency checks
# ---------------------------------------------------------------------------
from .preprocess import ensure_datasets_present
from .train import EnergyMeter, MissingSensorError

try:
    ensure_datasets_present(CONFIG, REPO_ROOT / "data")
except RuntimeError as exc:
    print(f"[ERROR] Dataset acquisition failed – {exc}")
    sys.exit(1)

try:
    _ = EnergyMeter.detect_available_backend()
except MissingSensorError as exc:
    print(f"[ERROR] {exc}")
    sys.exit(1)

if "ELECTRICITYMAP_TOKEN" not in os.environ:
    print(
        "[ERROR] Environment variable ELECTRICITYMAP_TOKEN missing – live grid-carbon data unavailable."
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
#  Import experiments and run sequentially
# ---------------------------------------------------------------------------
from .evaluate import (
    run_experiment1,
    run_experiment2,
    run_experiment3,
)

EXPERIMENTS = [
    ("exp1", run_experiment1),
    ("exp2", run_experiment2),
    ("exp3", run_experiment3),
]

for name, fn in EXPERIMENTS:
    description, result_json = fn(CONFIG, IMAGES_DIR)

    # ----------------------- Persist JSON ------------------------------
    out_file = RESULTS_DIR / f"{name}.json"
    out_file.write_text(json.dumps(result_json, indent=2))

    # -------------------- Console verification -------------------------
    print("\n" + "=" * 80)
    print(description)
    print("\nResults:")
    print(json.dumps(result_json, indent=2))
    print("Figures saved under .research/iteration6/images:")
    for fig in result_json.get("figures", []):
        print("  •", fig)
    print("=" * 80 + "\n")
