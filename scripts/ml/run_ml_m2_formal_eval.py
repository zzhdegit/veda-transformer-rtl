"""Evaluate the ML-M2 Formal best checkpoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.export.formal_export import evaluate_formal_checkpoint


def main() -> int:
    print(json.dumps(evaluate_formal_checkpoint(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
