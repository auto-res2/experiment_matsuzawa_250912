"""
evaluate.py – evaluation, statistics, and plotting utilities
────────────────────────────────────────────────────────────
The scientific evaluation logic for ReFuse-CL resides inside the upstream
`refuse_cl` package; therefore this file currently acts as a thin convenience
wrapper that re-exports those utilities if – and only if – the package is
available in the runtime environment.  This keeps the public interface stable
without duplicating code or violating the *no new logic* constraint.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("ReFuse-CL")

try:
    # Re-export for external callers
    from refuse_cl.evaluation import *  # noqa: F401,F403
except ModuleNotFoundError as err:  # pragma: no cover – optional dependency
    logger.warning(
        "Full evaluation suite not found (missing `refuse_cl` package). "
        "Only training-side features will be available.  Original error: %s",
        err,
    )

__all__: list[str] = []  # populated via wildcard import if available
