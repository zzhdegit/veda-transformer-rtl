#!/usr/bin/env python3
"""Stage 8B paper-array Python/model regression."""

import importlib.util
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(cmd):
    print("$ %s" % " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(ROOT))
    print("exit_code=%d" % completed.returncode)
    return completed.returncode


def run_python_test_fallback(test_file):
    spec = importlib.util.spec_from_file_location("_stage8b_tests", str(test_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tests = [name for name in dir(module) if name.startswith("test_")]
    for name in tests:
        getattr(module, name)()
    print("Fallback Python test runner: %d tests" % len(tests))
    print("PASS")
    return 0


def main():
    print("Stage 8B paper-structured PE array model regression")
    test_file = ROOT / "tb/model/test_stage8_paper_array.py"
    exit_code = run_python_test_fallback(test_file)
    if exit_code:
        return exit_code

    py_files = [
        "model/pe_array/__init__.py",
        "model/pe_array/paper_pe_reference.py",
        "model/pe_array/paper_pe_group_reference.py",
        "model/pe_array/paper_array_8x8x2_reference.py",
        "model/pe_array/paper_array_mapping.py",
        "model/pe_array/paper_array_cycle_model.py",
        "model/pe_array/paper_array_compare_legacy.py",
        "tb/model/test_stage8_paper_array.py",
    ]
    exit_code = run([sys.executable, "-m", "py_compile"] + py_files)
    if exit_code:
        return exit_code

    exit_code = run([sys.executable, "model/pe_array/paper_array_cycle_model.py"])
    if exit_code:
        return exit_code

    exit_code = run([sys.executable, "model/pe_array/paper_array_compare_legacy.py"])
    if exit_code:
        return exit_code

    print("Stage 8B paper-array model result: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
