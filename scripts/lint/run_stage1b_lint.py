import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/stage_01b/lint_results.txt"


RTL_FILES = [
    ROOT / "rtl/common/stream_reg.sv",
    ROOT / "rtl/arithmetic/fp16_to_fp32.sv",
    ROOT / "rtl/arithmetic/fp32_mac_wrapper.sv",
]


def find_dw_sim_dir():
    for candidate in [Path(p) for p in []]:
        if candidate.is_dir():
            return candidate
    for base in (Path("/usr/synopsys"),):
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
        if "$fatal" not in text:
            errors.append("missing fatal/assertion checks in %s" % path.relative_to(ROOT))

    for token in (
        "unsupported NaN/Inf input assertion failed",
        "no_unknown_output_when_valid",
        "data_stable_until_ready",
        "metadata_stable_until_ready",
    ):
        if token not in all_text:
            errors.append("missing assertion/stability token: %s" % token)
    return errors


def main():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    errors = static_hygiene()
    lines.append("Static Stage 1B RTL hygiene: %s" % ("FAIL" if errors else "PASS"))
    lines.extend(errors)

    vlogan = shutil.which("vlogan")
    if vlogan:
        out_dir = ROOT / "build" / "stage1b_vlogan_lint"
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
            ):
                dw_files.append(str(dw_dir / name))
        else:
            errors.append("DesignWare sim_ver directory not found for vlogan")

        cmd = [vlogan, "-full64", "-sverilog", "-work", "work"] + dw_files + [str(path) for path in RTL_FILES]
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
        lines.append("$ vlogan -full64 -sverilog -work work <DW sim files> <stage1b RTL files>")
        lines.append("vlogan_log=build/stage1b_vlogan_lint/vlogan.log")
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
