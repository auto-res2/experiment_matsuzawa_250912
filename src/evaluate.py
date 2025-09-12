"""
evaluate.py – Logging, metric handling and light-weight visualisation
utilities.  All functions are importable from other modules via
``from .evaluate import ...``
"""
from __future__ import annotations

import atexit
import json
from pathlib import Path
from typing import Any, Dict, Union, Optional

import matplotlib

matplotlib.use("Agg")  # Always use non-interactive backend
import matplotlib.pyplot as plt

# =============================================================================
# Metric logger (singleton) – persists results and echoes them to stdout
# =============================================================================

class MetricLogger:
    def __init__(self, out_file: Path):
        self._metrics: Dict[str, Any] = {}
        self._file = out_file
        atexit.register(self._dump)

    # ---------------------------------------------------------------------
    def log_metric(self, name: str, value: Any, tag: Union[str, int, None] = None):
        if tag is None:
            self._metrics[name] = value
        else:
            self._metrics.setdefault(name, {})[str(tag)] = value

    # ---------------------------------------------------------------------
    def _dump(self):
        # Ensure target directory exists
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as f:
            json.dump(self._metrics, f, indent=2)
        # Echo to stdout so that CI can pick it up immediately
        print("\n=======  Experiment Results  =======")
        print(json.dumps(self._metrics, indent=2))
        print("===================================\n")

# -----------------------------------------------------------------------------
# Global singleton helpers
# -----------------------------------------------------------------------------

LOGGER: Optional[MetricLogger] = None


def init_logger(out_path: Path):
    """Instantiate the global logger. Must be called exactly once."""
    global LOGGER
    if LOGGER is None:
        LOGGER = MetricLogger(out_path)
    else:
        raise RuntimeError("Logger already initialised – double initialisation")


def close_logger():
    """Flush to disk/stdout and reset the global logger singleton."""
    global LOGGER
    if LOGGER is not None:
        # Trigger explicit dump before discarding (atexit would also do it, but
        # we need immediate persistence so that subsequent experiment phases do
        # not overwrite the previous JSON).
        LOGGER._dump()
        LOGGER = None


def log_metric(name: str, value: Any, tag: Union[str, int, None] = None):
    if LOGGER is None:
        raise RuntimeError("Logger not initialised; call init_logger() first")
    LOGGER.log_metric(name, value, tag)


# =============================================================================
# Plotting helpers – All figures are saved under .research/iteration5/images
# =============================================================================

def _resolve_fig_path(filename: Path) -> Path:
    """Force all figure paths to comply with mandatory directory structure."""
    root = Path(".research/iteration5/images")
    root.mkdir(parents=True, exist_ok=True)
    # Keep only the stem provided by caller to avoid accidental directory
    # traversals while still making filenames unique.
    return root / f"{filename.stem}.pdf"


def save_line_plot(
    x,
    y,
    xlabel: str,
    ylabel: str,
    title: str,
    filename: Path,
):
    plt.figure()
    plt.plot(x, y, label=title, marker="o")
    for xi, yi in zip(x, y):
        plt.annotate(f"{yi:.3f}", (xi, yi))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    final_path = _resolve_fig_path(filename)
    plt.savefig(final_path, bbox_inches="tight")
    plt.close()
