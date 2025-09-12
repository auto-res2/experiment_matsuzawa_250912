"""
train.py – model-/hardware-specific utilities
This file merely re-houses logic from the original monolithic script – NO new
behaviour is introduced.
"""
from __future__ import annotations
import time
import contextlib
from typing import Optional

__all__ = [
    "MissingSensorError",
    "EnergyMeter",
]

# ---------------------------------------------------------------------------
#  Power / energy telemetry helpers (exact copy of original logic)
# ---------------------------------------------------------------------------
class MissingSensorError(RuntimeError):
    """Raised if the host has no readable power sensor."""
    pass

class _NVMLBackend:
    """Thin wrapper around NVIDIA NVML to get instantaneous power & energy."""
    def __init__(self):
        import pynvml  # deferred import – raises if library / driver absent
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    # Both methods are intentionally *tiny* to keep call-overhead negligible
    def power(self):
        import pynvml
        return pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1e3  # Watt

    def energy(self):
        import pynvml
        return (
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1e3  # Joule
        )


class EnergyMeter(contextlib.AbstractContextManager):
    """Context manager that records wall-clock time and consumed Joule."""

    def __init__(self, label: str = "device"):
        self.label = label
        self.backend = self.detect_available_backend()

    # ------------------------------------------------------------------
    @staticmethod
    def detect_available_backend():
        try:
            import pynvml  # noqa: F401 – import only for availability test
            return _NVMLBackend()
        except Exception as exc:
            raise MissingSensorError(
                "No NVML-capable GPU detected – energy telemetry unavailable."
            ) from exc

    # ------------------------------------------------------------------
    def __enter__(self):
        self._e0 = self.backend.energy()
        self._t0 = time.time()
        return self

    def __exit__(self, *exc):
        self._e1 = self.backend.energy()
        self._t1 = time.time()

    # ------------------------------------------------------------------
    def joules(self) -> float:
        return self._e1 - self._e0

    def wall_time(self) -> float:
        return self._t1 - self._t0
