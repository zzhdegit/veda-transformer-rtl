#!/usr/bin/env python3
"""Unified Stage 8 host tests."""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/stage_08/test_results.txt"


def run(cmd):
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.returncode, proc.stdout


def main():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    commands = [
        [sys.executable, "scripts/sim/run_stage8a_tests.py"],
        [sys.executable, "scripts/sim/run_stage8b_tests.py"],
        [sys.executable, "scripts/sim/run_stage8d_tests.py"],
        [sys.executable, "scripts/sim/run_stage7a_tests.py"],
    ]
    lines = []
    failures = 0
    for cmd in commands:
        code, output = run(cmd)
        lines.append("$ " + " ".join(cmd))
        lines.append(output.rstrip())
        lines.append("exit_code=%d" % code)
        lines.append("")
        if code != 0:
            failures += 1

    compile_paths = (
        list((ROOT / "model").rglob("*.py"))
        + list((ROOT / "tb").rglob("*.py"))
        + list((ROOT / "scripts").rglob("*.py"))
    )
    skipped_compile = []
    if sys.version_info < (3, 7):
        filtered = []
        for path in compile_paths:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(
                line.strip() == "from __future__ import annotations"
                for line in text.splitlines()
            ):
                skipped_compile.append(str(path.relative_to(ROOT)))
            else:
                filtered.append(path)
        compile_paths = filtered
    if skipped_compile:
        lines.append(
            "py_compile skipped Python 3.7+ files under Python %s.%s:"
            % sys.version_info[:2]
        )
        lines.extend(skipped_compile)
        lines.append("")
    cmd = [sys.executable, "-m", "py_compile"] + [str(path.relative_to(ROOT)) for path in compile_paths]
    code, output = run(cmd)
    lines.append("$ " + " ".join(cmd))
    lines.append(output.rstrip())
    lines.append("exit_code=%d" % code)
    lines.append("")
    if code != 0:
        failures += 1

    lines.append("result=%s" % ("FAIL" if failures else "PASS"))
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT.read_text(encoding="utf-8"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
