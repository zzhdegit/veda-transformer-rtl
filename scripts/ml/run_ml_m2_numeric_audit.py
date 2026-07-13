"""Run ML-M2 numeric audit tests and produce a JSON report."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> int:
    code = subprocess.call([sys.executable, "-m", "pytest", "ml/tests/test_training_numerics.py"])
    if code != 0:
        return code
    return subprocess.call([sys.executable, "-m", "ml.training.numeric_audit"])


if __name__ == "__main__":
    raise SystemExit(main())
