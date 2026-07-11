import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/stage_01/synth_check.txt"


def main() -> int:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Stage 1 synthesis check",
        "This script does not generate PPA. It only reports whether a real synthesis environment is visible.",
        "",
    ]

    dc_shell = shutil.which("dc_shell")
    target_lib = os.environ.get("TECH_LIB_ROOT")
    synopsys_root = os.environ.get("SYNOPSYS_ROOT")

    lines.append(f"dc_shell: {'found' if dc_shell else 'not found'}")
    lines.append(f"SYNOPSYS_ROOT: {'set' if synopsys_root else 'not set'}")
    lines.append(f"TECH_LIB_ROOT: {'set' if target_lib else 'not set'}")

    if not dc_shell:
        lines.append("DC elaboration: SKIPPED - dc_shell not found.")
        lines.append("No area, power, WNS, frequency, or process timing conclusion is produced.")
    else:
        work_dir = ROOT / "build" / "stage1_dc_run"
        work_dir.mkdir(parents=True, exist_ok=True)
        cmd = [dc_shell, "-f", str(ROOT / "scripts" / "synth" / "stage1_elaborate.tcl")]
        proc = subprocess.run(
            cmd, cwd=str(work_dir), universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        dc_log = ROOT / "reports" / "stage_01" / "dc_elaborate.log"
        dc_log.write_text(proc.stdout, encoding="utf-8", errors="ignore")
        lines.append("$ dc_shell -f scripts/synth/stage1_elaborate.tcl")
        lines.append("dc_exit_code=%d" % proc.returncode)
        if target_lib:
            lines.append("Target library root: set, but Stage 1 script did not produce PPA.")
        else:
            lines.append("Target library root: not set; DC run is analyze/elaborate/check_design only.")
        lines.append("DC log: reports/stage_01/dc_elaborate.log")
        if proc.returncode != 0:
            lines.append("DC elaboration result: FAIL")
        else:
            lines.append("DC elaboration result: PASS")
        lines.append("No area, power, WNS, frequency, or process timing conclusion is produced.")

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
