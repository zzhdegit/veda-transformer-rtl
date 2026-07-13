#!/usr/bin/env python3
"""Run ML-Q2 full-dataset benchmark training workflow."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.q2_benchmark import main


if __name__ == "__main__":
    main()
