#!/usr/bin/env python3
"""Run HW-H9-N1 Q2 length1 RTL validation."""

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main():
    script = ROOT / "scripts" / "sim" / "run_hw_h9_q2_length1_vcs.sh"
    result = subprocess.run(["bash", str(script)], cwd=str(ROOT))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
