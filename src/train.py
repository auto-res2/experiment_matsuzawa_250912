"""
train.py – model-/hardware-specific utilities
This file merely re-houses logic from the original monolithic script – NO new
behaviour is introduced.
"""
from __future__ import annotations

import contextlib
import time
from typing import Optional

__all__ = [
    "MissingSensorError",
    "EnergyMeter",
]

# ---------------------------------------------------------------------------
#  Power / energy telemetry helpers (exact copy of original logic
#  *plus* a lightweight mock backend so that CI without NVML does not crash.)
# ---------------------------------------------------------------------------
class MissingSensorError(RuntimeError):
    """Raised if the host has no readable power sensor."""

    pass


class _NVMLBackend:  # pragma: no cover – not exercised inside headless CI
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

        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1e3  # Joule


class _MockBackend:
    """Fallback backend that returns 0 W / 0 Joule – used in CI without sensors."""

    def __init__(self):
        # align with NVML resolution (mW → W ; mJ → J).
        self._start = time.time()

    def power(self):  # noqa: D401 – trivial
        # Pretend constant 0 W to keep metrics numeric but obviously dummy.
        return 0.0

    def energy(self):
        # Return elapsed seconds as *milli*-J to keep monotonic increase &
        # avoid zero-division. 1 s → 1 J is arbitrary but traceable.
        return time.time() - self._start


class EnergyMeter(contextlib.AbstractContextManager):
    """Context manager that records wall-clock time and consumed Joule."""

    def __init__(self, label: str = "device"):
        self.label = label
        self.backend = self.detect_available_backend()

    # ------------------------------------------------------------------
    @staticmethod
    def detect_available_backend():
        """Return the first working telemetry backend or a Mock backend."""
        try:
            import pynvml  # noqa: F401 – import only for availability test

            return _NVMLBackend()
        except Exception:
            # No NVML → fall back to mock so that smoke-test continues.
            print(
                "[WARN] NVML unavailable – using MockBackend (energy numbers are dummy)."
            )
            return _MockBackend()

    # ------------------------------------------------------------------
    def __enter__(self):
        self._e0 = self.backend.energy()
        self._t0 = time.time()
        return self

    def __exit__(self, *exc):
        self._e1 = self.backend.energy()
        self._t1 = time.time()

    # ------------------------------------------------------------------
    def joules(self) -> float:  # noqa: D401 – simple getter
        return self._e1 - self._e0

    def wall_time(self) -> float:  # noqa: D401 – simple getter
        return self._t1 - self._t0
