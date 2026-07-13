#!/usr/bin/env python3
"""Run Stage 8D paper-attention model tests and comparison reports."""

import importlib.util
import shutil
import subprocess
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/stage_08/phase_8d_model_tests.txt"


def run(cmd):
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.returncode, proc.stdout


def fallback_tests(test_files):
    failures = []
    executed = 0
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    for index, path in enumerate(test_files):
        spec = importlib.util.spec_from_file_location("_stage8d_test_%d" % index, path)
        if spec is None or spec.loader is None:
            failures.append((str(path.relative_to(ROOT)), "could not load module"))
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            for name, obj in vars(module).items():
                if name.startswith("test_") and callable(obj):
                    executed += 1
                    try:
                        obj()
                    except BaseException:
                        failures.append(("%s::%s" % (path.relative_to(ROOT), name), traceback.format_exc().rstrip()))
        except BaseException:
            failures.append((str(path.relative_to(ROOT)), traceback.format_exc().rstrip()))
    lines = ["Fallback Python test runner: %d tests" % executed]
    if failures:
        for name, detail in failures:
            lines.append("FAIL %s" % name)
            lines.append(detail)
        return 1, "\n".join(lines)
    lines.append("PASS")
    return 0, "\n".join(lines)


def main():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    test_files = [ROOT / "tb/model/test_stage8_paper_attention.py"]
    lines = []
    failures = 0

    if shutil.which("pytest") or run([sys.executable, "-m", "pytest", "--version"])[0] == 0:
        cmd = [sys.executable, "-m", "pytest"] + [str(path.relative_to(ROOT)) for path in test_files]
        code, output = run(cmd)
        lines.append("$ " + " ".join(cmd))
        lines.append(output.rstrip())
        lines.append("exit_code=%d" % code)
        lines.append("")
        if code != 0:
            failures += 1
    else:
        code, output = fallback_tests(test_files)
        lines.append("$ fallback-python-test-runner stage8d paper attention tests")
        lines.append(output)
        lines.append("exit_code=%d" % code)
        lines.append("")
        if code != 0:
            failures += 1

    compile_paths = [
        ROOT / "model/attention/paper_attention_reference.py",
        ROOT / "model/attention/paper_attention_cycle_model.py",
        ROOT / "tb/model/test_stage8_paper_attention.py",
    ]
    cmd = [sys.executable, "-m", "py_compile"] + [str(path.relative_to(ROOT)) for path in compile_paths]
    code, output = run(cmd)
    lines.append("$ " + " ".join(cmd))
    lines.append(output.rstrip())
    lines.append("exit_code=%d" % code)
    lines.append("")
    if code != 0:
        failures += 1

    for cmd in (
        [sys.executable, "model/attention/paper_attention_reference.py"],
        [sys.executable, "model/attention/paper_attention_cycle_model.py"],
    ):
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
