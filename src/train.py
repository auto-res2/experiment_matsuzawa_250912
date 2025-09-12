"""
train.py – Contains experiment super-classes and concrete experiment
implementations that (optionally) train models or run simulations.
All heavy-weight operations are skipped when cfg.get("smoke_test", False)
         is True so that the smoke-test finishes within a few seconds and
         without external resources.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any

from .preprocess import fetch_dataset, DataUnavailableError
from .evaluate import log_metric

# ==========================================================================
# Base experiment class
# ==========================================================================

class Experiment(ABC):
    """Common abstract base class for every experiment."""

    def __init__(self, cfg: Dict[str, Any], data_root: Path):
        self.cfg = cfg
        self.data_root = data_root
        self.smoke = bool(cfg.get("smoke_test", False))

    # ------------------------------------------------------------------
    @abstractmethod
    def run(self):
        """Execute the experiment. MUST call log_metric for every result."""


# ==========================================================================
# Concrete experiments originally shipped in the monolithic script
# ==========================================================================

class Align16Experiment(Experiment):
    """End-to-End Alignment Test-bed (simulation)."""

    name = "ALIGN-16"

    def run(self):
        print("Running ALIGN-16 experiment – End-to-End Alignment Test-bed\n")

        if self.smoke:
            # Fast path: just record that the experiment would run.
            log_metric("status", "skipped_in_smoke", tag=self.name)
            return

        # -----------------------------------------------------------------
        # 1. Ensure dataset availability (will raise if missing)
        fetch_dataset(
            "hiddenbias_cars",
            self.cfg["datasets"]["hiddenbias_cars"],
            self.data_root,
        )

        # NOTE: A full Omnet++ + timing-accurate simulation cannot be shipped
        # in this compact sample.  We *explicitly* abort so that callers see a
        # clean, policy-compliant error instead of half-baked placeholders.
        raise RuntimeError(
            "ALIGN-16 requires the proprietary Omnet++ simulation and the "
            "HiddenBias-Cars dataset, which are NOT accessible. Execution "
            "terminated as per STRICT NO-FALLBACK RULE."
        )


class HIL12Experiment(Experiment):
    """Hardware-in-the-Loop experiment."""

    name = "HIL-12"

    def run(self):
        print("Running HIL-12 Hardware-in-the-Loop experiment\n")

        if self.smoke:
            log_metric("status", "skipped_in_smoke", tag=self.name)
            return

        fetch_dataset(
            "cubesat_drift",
            self.cfg["datasets"]["cubesat_drift"],
            self.data_root,
        )
        raise RuntimeError(
            "HIL-12 requires physical MCU hardware and CubeSat-Drift trace, "
            "which are not publicly downloadable – aborting."
        )


class Face80Experiment(Experiment):
    """Causal & Intersectional Bias Audit."""

    name = "FACE-80"

    def run(self):
        print("Running FACE-80 Causal & Intersectional Bias Audit\n")

        if self.smoke:
            log_metric("status", "skipped_in_smoke", tag=self.name)
            return

        fetch_dataset(
            "intersect_faces",
            self.cfg["datasets"]["intersect_faces"],
            self.data_root,
        )
        raise RuntimeError(
            "FACE-80 requires the Intersect-Faces dataset (1 M images) which "
            "is not publicly accessible here – aborting execution."
        )
