"""src/evaluate.py – numerical summary & simple bar plot."""
from __future__ import annotations

import json
import pathlib
from typing import Any

import matplotlib

# Use a non-interactive backend suitable for headless servers
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def summarise_and_plot(json_path: pathlib.Path, image_dir: pathlib.Path) -> str:
    """Reads a JSON result file and produces a bar-plot (Average Accuracy)."""

    data: Any = json.loads(json_path.read_text())
    aa = [r["AA"] for r in data["runs"]]
    labels = [f"{r['dataset']}-s{r['seed']}" for r in data["runs"]]

    plt.figure(figsize=(8, 3))
    plt.bar(labels, aa, color="steelblue")
    for i, v in enumerate(aa):
        plt.text(i, v + 0.01, f"{v * 100:.1f}%", ha="center", va="bottom", fontsize=8)
    plt.ylabel("Average Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    image_dir.mkdir(parents=True, exist_ok=True)
    fig_name = image_dir / "accuracy_tiger_lite.pdf"
    plt.savefig(fig_name, bbox_inches="tight")
    plt.close()
    return str(fig_name)


# ---------------------------------------------------------------------------
# Hardware-in-the-loop demo stub
# ---------------------------------------------------------------------------

def run_mcu_demo(*_args, **_kwargs):  # noqa: D401, ANN001
    raise RuntimeError(
        "Hardware-in-the-loop demo requires a physical STM32H7 board – aborting."
    )
