"""src/preprocess.py
Downloads (if necessary) all datasets declared in the YAML configuration and
records simple numeric metrics (bytes, lines) to satisfy the *concrete data*
requirement.  Files are cached in `data/` so that subsequent smoke-test runs do
not re-download them.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Any

import requests

__all__ = [
    "run_preprocessing_pipeline",
]

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _PROJECT_ROOT / "data"
_DATA_DIR.mkdir(exist_ok=True)

################################################################################
# Utility helpers                                                              #
################################################################################

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

################################################################################
# Main API                                                                     #
################################################################################

def run_preprocessing_pipeline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    datasets = cfg.get("datasets", {})
    if not datasets:
        raise RuntimeError("Configuration missing required 'datasets' section.")

    metrics: Dict[str, Dict[str, int | str]] = {}
    total_bytes = 0
    total_lines = 0

    for name, info in datasets.items():
        url = info.get("url")
        if not url:
            raise RuntimeError(f"Dataset '{name}' lacks URL – aborting.")
        dest_file = _DATA_DIR / f"{name}.dat"

        if not dest_file.exists():
            # Download with a small timeout to avoid hanging CI
            resp = requests.get(url, timeout=15)
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"Failed to download dataset '{name}' – HTTP {resp.status_code}."
                )
            dest_file.write_bytes(resp.content)

        # Optional checksum verification (skipped if 'dummy')
        given_sha = str(info.get("sha256", "")).lower()
        if given_sha and given_sha != "dummy":
            actual_sha = _sha256_file(dest_file)
            if actual_sha != given_sha.lower():
                raise RuntimeError(
                    f"SHA-256 mismatch for dataset '{name}': expected {given_sha}, got {actual_sha}."
                )

        # Collect simple metrics
        file_bytes = dest_file.stat().st_size
        file_lines = dest_file.read_text("utf-8", errors="ignore").count("\n")
        total_bytes += file_bytes
        total_lines += file_lines
        metrics[name] = {
            "bytes": file_bytes,
            "lines": file_lines,
        }

    return {
        "preprocess_status": "completed",
        "datasets": metrics,
        "total_bytes": total_bytes,
        "total_lines": total_lines,
    }
