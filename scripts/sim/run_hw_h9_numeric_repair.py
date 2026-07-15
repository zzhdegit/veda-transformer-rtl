#!/usr/bin/env python3
"""Host wrapper for the HW-H9-N1 numeric repair VCS regression."""

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main():
    script = ROOT / "scripts" / "sim" / "run_hw_h9_numeric_repair_vcs.sh"
    proc = subprocess.run(["bash", str(script)], cwd=str(ROOT))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
