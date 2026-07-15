"""Compare ML-M3 RTL captures against the hardware-aware bit model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.cosim.m3_compare import compare_rtl_outputs


if __name__ == "__main__":
    print(json.dumps(compare_rtl_outputs(), indent=2, sort_keys=True))
