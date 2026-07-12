import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/stage_03/lint_results.txt"


RTL_FILES = [
    ROOT / "rtl/common/stream_reg.sv",
    ROOT / "rtl/memory/sram_2p_wrapper.sv",
    ROOT / "rtl/arithmetic/fp16_to_fp32.sv",
    ROOT / "rtl/arithmetic/fp32_mac_wrapper.sv",
    ROOT / "rtl/arithmetic/fp32_add_wrapper.sv",
    ROOT / "rtl/arithmetic/fp32_exp_wrapper.sv",
    ROOT / "rtl/arithmetic/fp32_recip_wrapper.sv",
    ROOT / "rtl/pe/lane_mask_generator.sv",
    ROOT / "rtl/pe/accumulator_bank.sv",
    ROOT / "rtl/pe/pe_perf_counter.sv",
    ROOT / "rtl/pe/pe_lane.sv",
    ROOT / "rtl/pe/fp32_reduction_tree.sv",
    ROOT / "rtl/pe/reconfigurable_pe_core.sv",
    ROOT / "rtl/attention/attention_score_scaler.sv",
    ROOT / "rtl/attention/score_buffer.sv",
    ROOT / "rtl/attention/softmax_reduction.sv",
    ROOT / "rtl/attention/softmax_normalization.sv",
    ROOT / "rtl/attention/single_head_attention_controller.sv",
    ROOT / "rtl/attention/single_head_attention.sv",
]


def find_dw_sim_dir():
    base = Path("/usr/synopsys")
    if base.is_dir():
        matches = list(base.glob("*/dw/sim_ver")) + list(base.glob("*/*/dw/sim_ver"))
        for match in matches:
            if match.is_dir() and (match / "vcs" / "DW_exp2.v").is_file():
                return match
        for match in matches:
            if match.is_dir():
                return match
    return None


def static_hygiene():
    errors = []
    all_text = ""
    wrapper_names = {
        "fp32_mac_wrapper.sv",
        "fp32_add_wrapper.sv",
        "fp32_exp_wrapper.sv",
        "fp32_recip_wrapper.sv",
    }
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
        if path.name not in wrapper_names and re.search(r"\bDW_fp_|\bDW_exp2\b", text):
            errors.append("DesignWare direct instance/reference outside wrapper: %s" % path.relative_to(ROOT))

    for token in (
        "no_start_while_busy",
        "score_write_count <= seq_len",
        "no_score_read_before_written",
        "no SV update before probability valid",
        "score index and V index match",
        "no buffer overflow",
        "output stable until ready",
        "metadata stable",
        "no_unknown_output_when_valid",
        "payload_stable_until_ready",
    ):
        if token not in all_text:
            errors.append("missing Stage 3 assertion/stability token: %s" % token)
    return errors


def main():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    errors = static_hygiene()
    lines.append("Static Stage 3 RTL hygiene: %s" % ("FAIL" if errors else "PASS"))
    lines.extend(errors)

    vlogan = shutil.which("vlogan")
    if vlogan:
        out_dir = ROOT / "build" / "stage3_vlogan_lint"
        out_dir.mkdir(parents=True, exist_ok=True)
        dw_dir = find_dw_sim_dir()
        dw_files = []
        if dw_dir is not None:
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
                "DW_exp2.v",
                "DW_fp_exp.v",
                "DW_fp_div.v",
            ):
                dw_files.append(str(dw_dir / name))
        else:
            errors.append("DesignWare sim_ver directory not found for vlogan")

        cmd = [vlogan, "-full64", "-sverilog", "-work", "work"]
        if dw_dir is not None:
            cmd.append("+incdir+%s" % str(dw_dir))
        cmd += dw_files + [str(path) for path in RTL_FILES]
        proc = subprocess.run(
            cmd,
            cwd=str(out_dir),
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (out_dir / "vlogan.log").write_text(proc.stdout, encoding="utf-8", errors="ignore")
        diagnostics = [
            line for line in proc.stdout.splitlines()
            if re.search(r"\b(Warning|Error|Fatal)\b", line, re.IGNORECASE)
        ]
        lines.append("$ vlogan -full64 -sverilog -work work <DW sim files> <stage3 RTL files>")
        lines.append("vlogan_log=build/stage3_vlogan_lint/vlogan.log")
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
