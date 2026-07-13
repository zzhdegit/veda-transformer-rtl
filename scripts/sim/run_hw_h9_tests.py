#!/usr/bin/env python3
"""Hardware Stage H9 host/model tests."""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/hw_h9/test_results.txt"


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
        [sys.executable, "-m", "pytest", "tb/model/test_hw_h9_interleaved_attention.py"],
        [sys.executable, "model/attention/paper_interleaved_cycle_model.py"],
        [sys.executable, "model/attention/paper_interleaved_compare_h8.py"],
        [sys.executable, "scripts/sim/run_stage8_tests.py"],
    ]
    lines = ["Hardware Stage H9 host/model test report", ""]
    failures = 0
    for cmd in commands:
        code, output = run(cmd)
        lines.append("$ " + " ".join(cmd))
        lines.append(output.rstrip())
        lines.append("exit_code=%d" % code)
        lines.append("")
        if code != 0:
            failures += 1

    compile_paths = [
        ROOT / "model/pe_array/paper_array_mapping.py",
        ROOT / "model/attention/paper_interleaved_attention_reference.py",
        ROOT / "model/attention/paper_interleaved_softmax_reference.py",
        ROOT / "model/attention/paper_interleaved_cycle_model.py",
        ROOT / "model/attention/paper_interleaved_compare_h8.py",
        ROOT / "tb/model/test_hw_h9_interleaved_attention.py",
        ROOT / "scripts/sim/run_hw_h9_tests.py",
        ROOT / "scripts/lint/run_hw_h9_lint.py",
        ROOT / "scripts/synth/run_hw_h9_synth_check.py",
    ]
    cmd = [sys.executable, "-m", "py_compile"] + [
        str(path.relative_to(ROOT)) for path in compile_paths
    ]
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
