#!/usr/bin/env python3
"""Stage 8C static and vlogan checks for the paper array RTL."""

import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/stage_08/phase_8c_lint_results.txt"

RTL_FILES = [
    ROOT / "rtl/common/stream_reg.sv",
    ROOT / "rtl/arithmetic/fp16_to_fp32.sv",
    ROOT / "rtl/arithmetic/fp32_mac_wrapper.sv",
    ROOT / "rtl/arithmetic/fp32_add_wrapper.sv",
    ROOT / "rtl/pe/paper/paper_pe_cell.sv",
    ROOT / "rtl/pe/paper/paper_l1_reduction.sv",
    ROOT / "rtl/pe/paper/paper_l2_reduction.sv",
    ROOT / "rtl/pe/paper/paper_pe_group.sv",
    ROOT / "rtl/pe/paper/paper_array_8x8x2.sv",
]


def find_dw_sim_dir():
    base = Path("/usr/synopsys")
    if base.is_dir():
        matches = list(base.glob("*/dw/sim_ver")) + list(base.glob("*/*/dw/sim_ver"))
        for match in matches:
            if match.is_dir():
                return match
    return None


def static_hygiene():
    errors = []
    all_text = ""
    for path in RTL_FILES:
        if not path.exists():
            errors.append("missing RTL file: %s" % path.relative_to(ROOT))
            continue
        text = path.read_text(encoding="utf-8")
        all_text += "\n" + text
        if re.search(r"\b(real|shortreal)\b", text):
            errors.append("forbidden real/shortreal in %s" % path.relative_to(ROOT))
        if "`default_nettype none" not in text:
            errors.append("missing default_nettype none in %s" % path.relative_to(ROOT))
        if path.name not in ("fp32_mac_wrapper.sv", "fp32_add_wrapper.sv") and re.search(r"\bDW_fp_", text):
            errors.append("DesignWare direct reference outside wrappers: %s" % path.relative_to(ROOT))

    required_tokens = [
        "array_has_exactly_128_pe_cells",
        "group_count_is_two",
        "row_count_is_eight",
        "column_count_is_eight",
        "pe_type_mapping_matches_spec",
        "command_stable_until_ready",
        "no_mode_switch_with_inflight_data",
        "output_stable_until_ready",
        "done_stable_until_ready",
        "no_unknown_output_when_valid",
        "l1_reduction_order_legal",
        "l2_reduction_order_legal",
    ]
    for token in required_tokens:
        if token not in all_text:
            errors.append("missing Stage 8C assertion token: %s" % token)

    group_text = (ROOT / "rtl/pe/paper/paper_pe_group.sv").read_text(encoding="utf-8")
    if len(re.findall(r"paper_pe_cell\s*#", group_text)) != 1:
        errors.append("paper_pe_group must instantiate paper_pe_cell inside generate")
    if "for (row_g = 0; row_g < 8" not in group_text or "for (col_g = 0; col_g < 8" not in group_text:
        errors.append("paper_pe_group must preserve explicit 8x8 generate loops")

    return errors


def main():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    errors = static_hygiene()
    lines = ["Static Stage 8C RTL hygiene: %s" % ("FAIL" if errors else "PASS")]
    lines.extend(errors)

    vlogan = shutil.which("vlogan")
    if vlogan:
        out_dir = ROOT / "build/stage8c_vlogan_lint"
        out_dir.mkdir(parents=True, exist_ok=True)
        dw_dir = find_dw_sim_dir()
        dw_files = []
        if dw_dir is None:
            errors.append("DesignWare sim_ver directory not found for vlogan")
        else:
            for name in (
                "DW_fp_addsub.v",
                "DW_fp_dp2.v",
                "DW_ifp_mult.v",
                "DW_ifp_addsub.v",
                "DW_fp_ifp_conv.v",
                "DW_ifp_fp_conv.v",
                "DW_fp_mult.v",
                "DW_fp_add.v",
                "DW_fp_mac.v",
            ):
                dw_files.append(str(dw_dir / name))
        cmd = [vlogan, "-full64", "-sverilog", "-work", "work"] + dw_files + [str(path) for path in RTL_FILES]
        proc = subprocess.run(cmd, cwd=str(out_dir), universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out_dir / "vlogan.log").write_text(proc.stdout, encoding="utf-8", errors="ignore")
        diagnostics = [line for line in proc.stdout.splitlines() if re.search(r"\b(Warning|Error|Fatal)\b", line, re.IGNORECASE)]
        lines.append("$ vlogan -full64 -sverilog -work work <DW sim files> <stage8c RTL files>")
        lines.append("vlogan_log=build/stage8c_vlogan_lint/vlogan.log")
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

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT.read_text(encoding="utf-8"))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
