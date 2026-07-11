import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/stage_01/lint_results.txt"


RTL_FILES = [
    ROOT / "rtl/common/stream_reg.sv",
    ROOT / "rtl/common/skid_buffer.sv",
    ROOT / "rtl/memory/sync_fifo.sv",
    ROOT / "rtl/memory/sram_1p_wrapper.sv",
    ROOT / "rtl/memory/sram_2p_wrapper.sv",
    ROOT / "rtl/arithmetic/mul_unit.sv",
    ROOT / "rtl/arithmetic/add_unit.sv",
    ROOT / "rtl/arithmetic/mac_unit.sv",
    ROOT / "rtl/arithmetic/compare_max.sv",
    ROOT / "rtl/arithmetic/round_sat.sv",
]


def static_hygiene():
    errors = []
    all_text = ""
    for path in RTL_FILES:
        if not path.exists():
            errors.append(f"missing RTL file: {path.relative_to(ROOT)}")
            continue
        text = path.read_text(encoding="utf-8")
        all_text += "\n" + text
        if re.search(r"\b(real|shortreal)\b", text):
            errors.append(f"forbidden real/shortreal in {path.relative_to(ROOT)}")
        if "`default_nettype none" not in text:
            errors.append(f"missing default_nettype none in {path.relative_to(ROOT)}")
        if "$fatal" not in text:
            errors.append(f"missing parameter fatal checks in {path.relative_to(ROOT)}")

    required_assertions = [
        "valid_stable_until_ready",
        "data_stable_until_ready",
        "metadata_stable_until_ready",
        "no_fifo_write_when_full",
        "no_fifo_read_when_empty",
        "fifo_occupancy_in_range",
        "transaction_count_conserved",
        "no_unknown_output_when_valid",
    ]
    for token in required_assertions:
        if token not in all_text:
            errors.append(f"missing assertion token: {token}")
    return errors


def main() -> int:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    errors = static_hygiene()

    if errors:
        lines.append("Static RTL hygiene: FAIL")
        lines.extend(errors)
    else:
        lines.append("Static RTL hygiene: PASS")

    vlogan = shutil.which("vlogan")
    if vlogan:
        out_dir = ROOT / "build" / "stage1_vlogan_lint"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [vlogan, "-full64", "-sverilog", "-work", "work"] + [
            str(path) for path in RTL_FILES
        ]
        proc = subprocess.run(
            cmd, cwd=str(out_dir), universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        vlogan_log = out_dir / "vlogan.log"
        vlogan_log.write_text(proc.stdout, encoding="utf-8", errors="ignore")
        diagnostics = [
            line for line in proc.stdout.splitlines()
            if re.search(r"\b(Warning|Error|Fatal)\b", line, re.IGNORECASE)
        ]
        lines.append("$ vlogan -full64 -sverilog -work work <stage1 RTL files>")
        lines.append("vlogan_log=build/stage1_vlogan_lint/vlogan.log")
        lines.append("vlogan_exit_code=%d" % proc.returncode)
        if diagnostics:
            lines.append("vlogan_diagnostics:")
            lines.extend(diagnostics)
        else:
            lines.append("vlogan_diagnostics: none")
        if proc.returncode != 0:
            errors.append("vlogan lint/compile failed")
    else:
        lines.append("VCS/vlogan lint compile: SKIPPED - vlogan executable not found.")

    verilator = shutil.which("verilator")
    if verilator:
        cmd = [verilator, "--lint-only", "--sv"] + [str(path) for path in RTL_FILES]
        proc = subprocess.run(
            cmd, cwd=str(ROOT / "build"), universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        lines.append("$ " + " ".join(cmd))
        lines.append(proc.stdout.rstrip())
        lines.append("verilator_exit_code=%d" % proc.returncode)
        if proc.returncode != 0:
            errors.append("verilator lint failed")
    else:
        lines.append("External RTL lint: SKIPPED - verilator executable not found.")

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT.read_text(encoding="utf-8"))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
