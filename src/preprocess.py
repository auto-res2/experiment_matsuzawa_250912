# src/preprocess.py
"""Minimal helper that returns tiny src/tgt sentence lists for smoke testing.

For the full experiment you would replace this with real dataset loading &
pre-processing.  The function signature purposefully mimics that of a realistic
loader so that the downstream evaluation code remains unchanged.
"""
from __future__ import annotations

from typing import List, Tuple


def load_demo_split() -> Tuple[List[str], List[str]]:
    """Return a *very* small parallel corpus for CI smoke tests."""
    src = [
        "Hello, how are you?",
        "The quick brown fox jumps over the lazy dog.",
    ]
    # extremely rough German translations – good enough for unit testing
    tgt = [
        "Hallo, wie geht es dir?",
        "Der schnelle braune Fuchs springt über den faulen Hund.",
    ]
    return src, tgt
