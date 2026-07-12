import shutil
import subprocess
import sys
import traceback
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/stage_03/test_results.txt"


def run(cmd):
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.returncode, proc.stdout


def run_python_test_fallback(test_files):
    failures = []
    executed = 0

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    for index, path in enumerate(test_files):
        module_name = "_stage3_fallback_%d_%s" % (index, path.stem)
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            failures.append((str(path.relative_to(ROOT)), "could not load module"))
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            tests = [
                (name, obj)
                for name, obj in vars(module).items()
                if name.startswith("test_") and callable(obj)
            ]
            for name, test in tests:
                executed += 1
                try:
                    test()
                except BaseException:
                    failures.append((
                        "%s::%s" % (path.relative_to(ROOT), name),
                        traceback.format_exc().rstrip(),
                    ))
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
    lines = []
    failures = 0

    test_files = [
        ROOT / "tb/model/test_stage1b_fp.py",
        ROOT / "tb/model/test_stage2_pe.py",
        ROOT / "tb/model/test_stage3_attention.py",
    ]

    if shutil.which("pytest") or run([sys.executable, "-m", "pytest", "--version"])[0] == 0:
        commands = [[sys.executable, "-m", "pytest"] + [str(path.relative_to(ROOT)) for path in test_files]]
    else:
        commands = []
        code, output = run_python_test_fallback(test_files)
        lines.append("$ fallback-python-test-runner stage1b/stage2/stage3 model tests")
        lines.append(output)
        lines.append("exit_code=%d" % code)
        lines.append("")
        if code != 0:
            failures += 1

    commands.append(
        [sys.executable, "-m", "py_compile"]
        + [str(path.relative_to(ROOT)) for path in (ROOT / "model").rglob("*.py")]
        + [str(path.relative_to(ROOT)) for path in (ROOT / "tb").rglob("*.py")]
        + [str(path.relative_to(ROOT)) for path in (ROOT / "scripts").rglob("*.py")]
    )

    for cmd in commands:
        code, output = run(cmd)
        lines.append("$ " + " ".join(cmd))
        lines.append(output.rstrip())
        lines.append("exit_code=%d" % code)
        lines.append("")
        if code != 0:
            failures += 1

    if shutil.which("vcs"):
        cmd = ["bash", str(ROOT / "scripts" / "sim" / "run_stage3_vcs.sh")]
        code, output = run(cmd)
        lines.append("$ " + " ".join(cmd))
        lines.append(output.rstrip())
        lines.append("exit_code=%d" % code)
        if code != 0:
            failures += 1
    else:
        lines.append("Stage 3 RTL simulation: SKIPPED - vcs executable not found on host.")

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT.read_text(encoding="utf-8"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

