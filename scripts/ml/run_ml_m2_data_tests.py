"""Run ML-M2 data/tokenizer tests."""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    return subprocess.call([sys.executable, "-m", "pytest", "ml/tests/test_data_tokenizer.py"])


if __name__ == "__main__":
    raise SystemExit(main())

