"""Run the ML-M3 Q2 artifact and weight-mapping audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.cosim.m3_artifact_audit import run_artifact_audit


if __name__ == "__main__":
    print(json.dumps(run_artifact_audit(), indent=2, sort_keys=True))
