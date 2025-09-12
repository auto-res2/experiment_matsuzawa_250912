# src/main.py
"""Command-line entry-point that orchestrates the end-to-end experiment.

Usage:
    python -m src.main --smoke-test       # quick CI run
    python -m src.main --full-experiment  # full benchmark (may take hours)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from .config import ExperimentConfig
from .evaluate import evaluate_translation, bar_plot
from .preprocess import load_demo_split
from .train import build_super_surrogate

# ---------------------------------------------------------------------------
# CLI & config handling
# ---------------------------------------------------------------------------

_CONFIG_ROOT = Path("config")


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QuADRoN-DM experiment runner")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke-test", action="store_true", help="Run quick CI smoke test")
    g.add_argument("--full-experiment", action="store_true", help="Run full benchmark")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# main orchestration
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    cfg_path = _CONFIG_ROOT / ("smoke_test.yaml" if args.smoke_test else "full_experiment.yaml")
    raw_cfg = _load_yaml(cfg_path)
    cfg = ExperimentConfig.from_dict(raw_cfg)

    # ------------------------------------------------------------------
    # build model & run evaluation
    # ------------------------------------------------------------------
    model = build_super_surrogate(cfg.model_name)

    # Very small dataset for smoke test; for full experiment you'd integrate a
    # real loader that respects cfg.dataset_* entries.
    src, tgt = load_demo_split()

    metrics = evaluate_translation(model, src, tgt, cfg)

    # Optional: a tiny plot – harmless in CI once matplotlib is installed
    bar_plot({cfg.experiment_name: metrics}, "bleu", "BLEU on demo set", cfg.experiment_name)


# ---------------------------------------------------------------------------
# entry-point wrapper for pyproject-based invocation
# ---------------------------------------------------------------------------

def cli_entry() -> None:  # pragma: no cover – trivial wrapper
    try:
        main()
    except Exception as exc:  # fail-fast with a clear message
        sys.stderr.write(f"Fatal error – experiment aborted:\n{exc}\n")
        sys.exit(1)


if __name__ == "__main__":
    cli_entry()
