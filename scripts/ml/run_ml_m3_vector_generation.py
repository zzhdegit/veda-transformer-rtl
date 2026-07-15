"""Generate ML-M3 real-input vectors from the frozen Q2 checkpoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.cosim.m3_vector_export import generate_vectors


if __name__ == "__main__":
    print(json.dumps(generate_vectors(), indent=2, sort_keys=True))
