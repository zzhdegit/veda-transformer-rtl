"""Small RTL vector fixture builder for ML-M2 exports."""

from __future__ import annotations

import json
from pathlib import Path


def write_small_rtl_fixture(path: str | Path, trace_manifest: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "ML-M2F",
        "description": "Small software-generated RTL fixture manifest; no large vectors committed.",
        "trace_manifest": trace_manifest,
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

