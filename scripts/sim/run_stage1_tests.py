import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/stage_01/test_results.txt"


def run(cmd):
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.returncode, proc.stdout


def main() -> int:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    failures = 0

    commands = []
    if shutil.which("pytest") or run([sys.executable, "-m", "pytest", "--version"])[0] == 0:
        commands.append([sys.executable, "-m", "pytest", "tb/model", "tb/unit"])
    else:
        lines.append("Python pytest regression: SKIPPED - pytest is not available for this interpreter.")
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
        lines.append(f"exit_code={code}")
        lines.append("")
        if code != 0:
            failures += 1

    vcs = shutil.which("vcs")
    if vcs is None:
        lines.append("RTL simulation: SKIPPED - vcs executable not found.")
    else:
        cmd = ["bash", str(ROOT / "scripts" / "sim" / "run_stage1_vcs.sh")]
        code, output = run(cmd)
        lines.append("$ " + " ".join(cmd))
        lines.append(output.rstrip())
        lines.append("exit_code=%d" % code)
        if code != 0:
            failures += 1

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT.read_text(encoding="utf-8"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
