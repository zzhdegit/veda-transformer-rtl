import os
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/stage_07/phase_7c_synth_check.txt"


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


def main():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Stage 7C synthesis/elaboration check",
        "This is analyze/elaborate/link/check_design only. It does not generate PPA.",
        "",
    ]
    errors = []
    dc_shell = shutil.which("dc_shell")
    dw_sldb = find_dw_sldb()
    target_lib = os.environ.get("TECH_LIB_ROOT")
    lines.append("dc_shell: %s" % ("found" if dc_shell else "not found"))
    lines.append("DW_FOUNDATION_SLDB: %s" % ("found" if dw_sldb else "not found"))
    lines.append("TECH_LIB_ROOT: %s" % ("set" if target_lib else "not set"))
    if not dc_shell:
        lines.append("DC elaboration: SKIPPED - dc_shell not found.")
        errors.append("dc_shell not found")
    elif not dw_sldb:
        lines.append("DC elaboration: SKIPPED - DesignWare foundation library not found.")
        errors.append("DW foundation library not found")
    else:
        work_dir = ROOT / "build" / "stage7c_dc_run"
        work_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["DW_FOUNDATION_SLDB"] = dw_sldb
        cmd = [dc_shell, "-f", str(ROOT / "scripts" / "synth" / "stage7c_elaborate.tcl")]
        proc = subprocess.run(
            cmd,
            cwd=str(work_dir),
            env=env,
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        dc_log = ROOT / "reports" / "stage_07" / "phase_7c_dc_elaborate.log"
        dc_log.write_text(proc.stdout, encoding="utf-8", errors="ignore")
        dc_errors = [
            line for line in proc.stdout.splitlines()
            if re.search(r"^(Error:|\*\*\* .*errors?)", line)
        ]
        lines.append("$ dc_shell -f scripts/synth/stage7c_elaborate.tcl")
        lines.append("dc_exit_code=%d" % proc.returncode)
        lines.append("Target library root: %s" % ("set, but no PPA generated" if target_lib else "not set"))
        lines.append("DC log: reports/stage_07/phase_7c_dc_elaborate.log")
        if proc.returncode != 0 or dc_errors:
            lines.append("DC elaboration result: FAIL")
            if dc_errors:
                lines.append("DC errors:")
                lines.extend(dc_errors[:40])
            errors.append("dc_shell elaboration failed")
        else:
            lines.append("DC elaboration result: PASS")
        lines.append("Checked ffn_engine for D_MODEL 8 and 16.")
        lines.append("No area, power, WNS, frequency, or process timing conclusion is produced.")

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT.read_text(encoding="utf-8"))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
