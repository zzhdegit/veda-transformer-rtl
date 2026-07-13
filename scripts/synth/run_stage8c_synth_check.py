#!/usr/bin/env python3
"""Stage 8C DC analyze/elaborate/link/check_design runner."""

import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/stage_08/phase_8c_synth_check.txt"


def find_dw_sldb():
    env_path = os.environ.get("DW_FOUNDATION_SLDB")
    if env_path and Path(env_path).is_file():
        return env_path
    base = Path("/usr/synopsys")
    if base.is_dir():
        matches = list(base.glob("*/libraries/syn/dw_foundation.sldb")) + list(base.glob("*/*/libraries/syn/dw_foundation.sldb"))
        for match in matches:
            if match.is_file():
                return str(match)
    return None


def main():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Stage 8C synthesis/elaboration check",
        "This is analyze/elaborate/link/check_design only. It does not generate PPA.",
        "",
    ]
    errors = []
    dc_shell = shutil.which("dc_shell")
    dw_sldb = find_dw_sldb()
    lines.append("dc_shell: %s" % ("found" if dc_shell else "not found"))
    lines.append("DW_FOUNDATION_SLDB: %s" % ("found" if dw_sldb else "not found"))
    lines.append("TECH_LIB_ROOT: %s" % ("set" if os.environ.get("TECH_LIB_ROOT") else "not set"))

    if not dc_shell:
        lines.append("DC elaboration: SKIPPED - dc_shell not found.")
        errors.append("dc_shell not found")
    elif not dw_sldb:
        lines.append("DC elaboration: SKIPPED - DesignWare foundation library not found.")
        errors.append("DW foundation library not found")
    else:
        work_dir = ROOT / "build/stage8c_dc_run"
        work_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["DW_FOUNDATION_SLDB"] = dw_sldb
        cmd = [dc_shell, "-f", str(ROOT / "scripts/synth/stage8c_elaborate.tcl")]
        proc = subprocess.run(cmd, cwd=str(work_dir), env=env, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (ROOT / "reports/stage_08/phase_8c_dc_elaborate.log").write_text(proc.stdout, encoding="utf-8", errors="ignore")
        lines.append("$ dc_shell -f scripts/synth/stage8c_elaborate.tcl")
        lines.append("dc_exit_code=%d" % proc.returncode)
        lines.append("DC log: reports/stage_08/phase_8c_dc_elaborate.log")
        if proc.returncode != 0:
            lines.append("DC elaboration result: FAIL")
            errors.append("dc_shell elaboration failed")
        else:
            lines.append("DC elaboration result: PASS")
        lines.append("Checked paper_array_8x8x2 hierarchy and check_design.")
        lines.append("No area, power, WNS, frequency, or process timing conclusion is produced.")

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT.read_text(encoding="utf-8"))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
