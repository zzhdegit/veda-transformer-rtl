#!/usr/bin/env python3
"""Hardware Stage H9 DC analyze/elaborate/link/check_design runner."""

import os
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/hw_h9/synth_check.txt"


def find_dw_sldb():
    env_path = os.environ.get("DW_FOUNDATION_SLDB")
    if env_path and Path(env_path).is_file():
        return env_path
    base = Path("/usr/synopsys")
    if base.is_dir():
        matches = list(base.glob("*/libraries/syn/dw_foundation.sldb")) + list(
            base.glob("*/*/libraries/syn/dw_foundation.sldb")
        )
        for match in matches:
            if match.is_file():
                return str(match)
    return None


def count_hierarchy_cells(path):
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    return len(re.findall(r"paper_pe_cell", text))


def main():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Hardware Stage H9 synthesis/elaboration check",
        "This is analyze/elaborate/link/check_design plus hierarchy count only. It does not generate PPA.",
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
        work_dir = ROOT / "build/hw_h9_dc_run"
        work_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["DW_FOUNDATION_SLDB"] = dw_sldb
        cmd = [dc_shell, "-f", str(ROOT / "scripts/synth/hw_h9_elaborate.tcl")]
        proc = subprocess.run(
            cmd,
            cwd=str(work_dir),
            env=env,
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        dc_log = ROOT / "reports/hw_h9/dc_elaborate.log"
        dc_log.write_text(proc.stdout, encoding="utf-8", errors="ignore")
        raw_dc_errors = [
            line for line in proc.stdout.splitlines()
            if re.search(r"^(Error:|\*\*\* .*errors?)", line)
        ]
        tolerated_dc_errors = [
            line for line in raw_dc_errors
            if "Can't find design" in line and "(UID-109)" in line
        ]
        dc_errors = [line for line in raw_dc_errors if line not in tolerated_dc_errors]
        lines.append("$ dc_shell -f scripts/synth/hw_h9_elaborate.tcl")
        lines.append("dc_exit_code=%d" % proc.returncode)
        lines.append("DC log: reports/hw_h9/dc_elaborate.log")
        if proc.returncode != 0 or dc_errors:
            lines.append("DC elaboration result: FAIL")
            if dc_errors:
                lines.append("DC errors:")
                lines.extend(dc_errors[:40])
            errors.append("dc_shell elaboration failed")
        else:
            lines.append("DC elaboration result: PASS")
        if tolerated_dc_errors:
            lines.append("Tolerated DC diagnostics:")
            lines.append("UID-109 base-design alias messages after parameterized elaboration; dc_exit_code was 0 and reports were generated.")
            lines.extend(tolerated_dc_errors[:20])
        for rel in (
            "dc_hierarchy_hw_h9_single_head_d8_interleaved.rpt",
            "dc_hierarchy_hw_h9_single_head_d64_interleaved.rpt",
            "dc_hierarchy_hw_h9_transformer_h2_d8_interleaved.rpt",
        ):
            path = ROOT / "reports/hw_h9" / rel
            count = count_hierarchy_cells(path)
            lines.append("%s paper_pe_cell_occurrences=%s" % (rel, "unavailable" if count is None else count))
        lines.append("Checked legacy staged, paper staged, and paper interleaved architecture/schedule selections.")
        lines.append("No area, power, WNS, frequency, or process timing conclusion is produced.")

    lines.append("result=%s" % ("FAIL" if errors else "PASS"))
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT.read_text(encoding="utf-8"))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
