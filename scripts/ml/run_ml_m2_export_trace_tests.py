"""Run ML-M2 export and trace tests."""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    return subprocess.call([sys.executable, "-m", "pytest", "ml/tests/test_export_trace.py"])


if __name__ == "__main__":
    raise SystemExit(main())

