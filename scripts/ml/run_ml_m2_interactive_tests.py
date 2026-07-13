"""Run ML-M2 post-acceptance interactive evaluation tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> int:
    commands = [
        [
            sys.executable,
            "-m",
            "py_compile",
            "ml/inference/interactive.py",
            "scripts/ml/run_ml_m2_chat.py",
            "scripts/ml/run_ml_m2_next_token.py",
            "scripts/ml/run_ml_m2_inspect.py",
            "scripts/ml/run_ml_m2_prompt_suite.py",
        ],
        [sys.executable, "-m", "pytest", "ml/tests/test_interactive_eval.py"],
    ]
    for cmd in commands:
        print("$ " + " ".join(cmd))
        code = subprocess.call(cmd)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
