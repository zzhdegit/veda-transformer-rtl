import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/stage_06/synth_check.txt"


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
        [sys.executable, "scripts/synth/run_stage6b_synth_check.py"],
        [sys.executable, "scripts/synth/run_stage6c_synth_check.py"],
        [sys.executable, "scripts/synth/run_stage6d_synth_check.py"],
        [sys.executable, "scripts/synth/run_stage6e_synth_check.py"],
    ]
    lines = [
        "Stage 6 final synthesis/elaboration regression",
        "This is analyze/elaborate/link/check_design only. It does not generate PPA.",
        "",
    ]
    failures = 0
    for cmd in commands:
        code, output = run(cmd)
        lines.append("$ " + " ".join(cmd))
        lines.append(output.rstrip())
        lines.append("exit_code=%d" % code)
        lines.append("")
        if code != 0:
            failures += 1

    lines.append("Stage 6 synthesis/elaboration result: %s" % ("FAIL" if failures else "PASS"))
    lines.append("No formal area, power, frequency, WNS, STA, or layout result is reported.")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT.read_text(encoding="utf-8"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
