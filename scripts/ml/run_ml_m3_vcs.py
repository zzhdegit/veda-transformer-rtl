"""Run ML-M3 real-weight transformer_layer VCS simulations."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.cosim.m3_trace_schema import ARTIFACT_ROOT, VECTOR_LENGTHS_REQUIRED, sha256_file, write_json


SCHEDULES = {"staged": 0, "interleaved": 1}
STALL_MODES = {"none": 0, "output": 1, "done": 2, "output_done": 3}
CONTAINER = "nailong"
CONTAINER_HW_ROOT = "/workspace/VEDA"
CONTAINER_MODEL_ROOT = "/workspace/VEDA_ml_m2"
CONTAINER_ARTIFACT_ROOT = "/workspace/VEDA_artifacts/ml_m3"


RTL_FILES = [
    "rtl/common/stream_reg.sv",
    "rtl/memory/sram_2p_wrapper.sv",
    "rtl/arithmetic/fp16_to_fp32.sv",
    "rtl/arithmetic/fp32_mac_wrapper.sv",
    "rtl/arithmetic/fp32_add_wrapper.sv",
    "rtl/arithmetic/fp32_exp_wrapper.sv",
    "rtl/arithmetic/fp32_recip_wrapper.sv",
    "rtl/arithmetic/fp32_sqrt_wrapper.sv",
    "rtl/arithmetic/fp32_to_fp16.sv",
    "rtl/pe/lane_mask_generator.sv",
    "rtl/pe/accumulator_bank.sv",
    "rtl/pe/pe_perf_counter.sv",
    "rtl/pe/pe_lane.sv",
    "rtl/pe/fp32_reduction_tree.sv",
    "rtl/pe/reconfigurable_pe_core.sv",
    "rtl/pe/paper/paper_pe_cell.sv",
    "rtl/pe/paper/paper_l1_reduction.sv",
    "rtl/pe/paper/paper_l2_reduction.sv",
    "rtl/pe/paper/paper_pe_group.sv",
    "rtl/pe/paper/paper_array_8x8x2.sv",
    "rtl/attention/paper/interleaved/paper_score_packet_pkg.sv",
    "rtl/attention/paper/interleaved/paper_score_buffer.sv",
    "rtl/attention/paper/interleaved/paper_probability_fifo.sv",
    "rtl/attention/paper/interleaved/paper_interleaved_attention_datapath.sv",
    "rtl/attention/paper/paper_attention_adapter.sv",
    "rtl/attention/attention_score_scaler.sv",
    "rtl/attention/score_buffer.sv",
    "rtl/attention/softmax_reduction.sv",
    "rtl/attention/softmax_normalization.sv",
    "rtl/attention/single_head_attention_controller.sv",
    "rtl/attention/single_head_attention.sv",
    "rtl/cache/multi_head_kv_cache_manager.sv",
    "rtl/cache/multi_head_generation_controller.sv",
    "rtl/attention/multi_head_generation_engine.sv",
    "rtl/projection/projection_input_buffer.sv",
    "rtl/projection/projection_weight_buffer.sv",
    "rtl/projection/shared_gemv_projection_core.sv",
    "rtl/projection/projection_controller.sv",
    "rtl/projection/qkv_staging_buffer.sv",
    "rtl/projection/concat_fp16_buffer.sv",
    "rtl/projection/head_concat_quantizer.sv",
    "rtl/projection/output_projection_controller.sv",
    "rtl/attention/projection_integrated_mha.sv",
    "rtl/transformer/rmsnorm_engine.sv",
    "rtl/transformer/residual_add_engine.sv",
    "rtl/transformer/ffn_engine.sv",
    "rtl/transformer/transformer_layer.sv",
]

DW_FILES = [
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
    "DW_fp_sqrt.v",
]


def _container_path(host_path: Path) -> str:
    text = str(host_path).replace("\\", "/")
    if text.startswith("D:/IC_Workspace/"):
        return "/workspace/" + text[len("D:/IC_Workspace/") :]
    raise ValueError(f"path is outside D:/IC_Workspace: {host_path}")


def _run_container_bash(script: str, timeout_seconds: int = 3600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "exec", CONTAINER, "bash", "-lc", script],
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_seconds,
    )


def docker_environment_audit() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    inspect = subprocess.run(
        ["docker", "inspect", CONTAINER],
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    checks["docker_inspect_exit_code"] = inspect.returncode
    checks["docker_inspect_available"] = inspect.returncode == 0
    for name, command in {
        "pwd": "pwd",
        "workspace_ls": "ls /workspace",
        "vcs": "command -v vcs && vcs -ID 2>&1 | sed -n '1p'",
        "vlogan": "command -v vlogan && vlogan -ID 2>&1 | sed -n '1p'",
        "hardware_head_file": "cat /workspace/VEDA/.git/HEAD 2>/dev/null || true",
        "hardware_head_commit": (
            "cd /workspace/VEDA && head=$(cat .git/HEAD 2>/dev/null || true); "
            "case \"$head\" in ref:*) ref=${head#ref: }; "
            "cat .git/$ref 2>/dev/null || awk -v r=\"$ref\" '$2==r{print $1}' .git/packed-refs;; "
            "*) echo \"$head\";; esac"
        ),
    }.items():
        out = _run_container_bash(command, timeout_seconds=120)
        checks[name] = {"exit_code": out.returncode, "output": out.stdout.strip()}
    write_json(ARTIFACT_ROOT / "manifests" / "docker_environment.json", checks)
    return checks


def _compile_script(schedule_name: str, schedule_value: int, build_dir: str, compile_log: str, simv: str) -> str:
    rtl_args = " ".join(f'"{CONTAINER_HW_ROOT}/{path}"' for path in RTL_FILES)
    dw_args = " ".join(f'"$DW_SIM_DIR/{name}"' for name in DW_FILES)
    tb = f"{CONTAINER_MODEL_ROOT}/ml/cosim/rtl_tb/tb_ml_m3_transformer_layer.sv"
    return f"""
set -u
find_dw_sim_dir() {{
  if [ -n "${{DW_SIM_DIR:-}}" ] && [ -d "$DW_SIM_DIR" ]; then echo "$DW_SIM_DIR"; return 0; fi
  for candidate in /usr/synopsys/*/dw/sim_ver /usr/synopsys/*/*/dw/sim_ver; do
    if [ -d "$candidate" ] && [ -f "$candidate/vcs/DW_exp2.v" ]; then echo "$candidate"; return 0; fi
  done
  return 1
}}
DW_SIM_DIR=$(find_dw_sim_dir) || exit 11
mkdir -p "{build_dir}"
cd "{build_dir}"
vcs -full64 -sverilog -debug_access+pp -assert svaext -timescale=1ns/1ps \\
  +incdir+"$DW_SIM_DIR" \\
  +define+ML_M3_N_HEAD=8 \\
  +define+ML_M3_D_HEAD=8 \\
  +define+ML_M3_MAX_SEQ_LEN=128 \\
  +define+ML_M3_ATTENTION_PE_ARCH=1 \\
  +define+ML_M3_ATTENTION_SCHEDULE={schedule_value} \\
  -Mdir="{build_dir}/{schedule_name}_csrc" \\
  -o "{simv}" \\
  {dw_args} \\
  {rtl_args} \\
  "{tb}" \\
  -top tb_ml_m3_transformer_layer \\
  -l "{compile_log}"
"""


def _run_script(simv: str, vector_path: str, capture_path: str, run_log: str, timeout_seconds: int) -> str:
    return _run_script_with_options(simv, vector_path, capture_path, "", run_log, timeout_seconds, False)


def _run_script_with_options(
    simv: str,
    vector_path: str,
    capture_path: str,
    node_path: str,
    run_log: str,
    timeout_seconds: int,
    diagnostic: bool,
    stall_mode_value: int,
) -> str:
    diagnostic_arg = "+ML_M3_DIAGNOSTIC" if diagnostic else ""
    node_arg = f'+ML_M3_NODE_FILE="{node_path}"' if node_path else ""
    return f"""
set -u
timeout {timeout_seconds}s "{simv}" \\
  +ML_M3_VECTOR_FILE="{vector_path}" \\
  +ML_M3_OUTPUT_FILE="{capture_path}" \\
  +ML_M3_STALL_MODE={stall_mode_value} \\
  {diagnostic_arg} \\
  {node_arg} \\
  -l "{run_log}"
"""


def _parse_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    fail_markers = re.findall(r"(ML_M3_TB_FAIL:.*|CHECK_FAIL.*|Fatal:.*|Error:.*|assert.*failed.*)", text)
    pass_line = None
    perf_line = None
    boundary_line = None
    token_lines = []
    for line in text.splitlines():
        if line.startswith("ML_M3_RTL_PASS"):
            pass_line = line
        elif line.startswith("ML_M3_PERF"):
            perf_line = line
        elif line.startswith("ML_M3_BOUNDARY_OBS"):
            boundary_line = line
        elif line.startswith("ML_M3_TOKEN_PASS"):
            token_lines.append(line)
        elif line.startswith("ML_M3_RTL_DIAGNOSTIC_DONE"):
            pass_line = line
    return {
        "pass": bool(pass_line) and not fail_markers and not pass_line.startswith("ML_M3_RTL_DIAGNOSTIC_DONE"),
        "pass_line": pass_line,
        "perf_line": perf_line,
        "boundary_line": boundary_line,
        "token_lines": token_lines,
        "fail_markers": fail_markers[:40],
    }


def run_vcs(
    lengths: list[int],
    schedules: list[str],
    run_id: str | None = None,
    diagnostic: bool = False,
    stall_modes: list[str] | None = None,
) -> dict[str, Any]:
    run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
    stall_modes = stall_modes or ["none"]
    (ARTIFACT_ROOT / "rtl_logs").mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / "temporary_build").mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / "traces").mkdir(parents=True, exist_ok=True)
    env = docker_environment_audit()
    results: dict[str, Any] = {
        "stage": "ML-M3",
        "run_id": run_id,
        "stall_modes": stall_modes,
        "docker_environment": env,
        "schedules": {},
    }
    for schedule_name in schedules:
        schedule_value = SCHEDULES[schedule_name]
        build_dir = f"{CONTAINER_ARTIFACT_ROOT}/temporary_build/vcs_{run_id}_{schedule_name}"
        simv = f"{build_dir}/ml_m3_{schedule_name}_simv"
        compile_log_host = ARTIFACT_ROOT / "rtl_logs" / f"ml_m3_{schedule_name}_compile_{run_id}.log"
        compile_log = _container_path(compile_log_host)
        compile = _run_container_bash(_compile_script(schedule_name, schedule_value, build_dir, compile_log, simv), timeout_seconds=3600)
        schedule_result: dict[str, Any] = {
            "schedule": schedule_name,
            "schedule_value": schedule_value,
            "compile_exit_code": compile.returncode,
            "compile_log": str(compile_log_host),
            "compile_stdout": compile.stdout[-8000:],
            "cases": {},
        }
        if compile.returncode != 0:
            schedule_result["result"] = "COMPILE_FAIL"
            results["schedules"][schedule_name] = schedule_result
            continue
        for length in lengths:
            vector_host = ARTIFACT_ROOT / "vectors" / f"len_{length}" / f"case_len_{length}.mem"
            for stall_mode_name in stall_modes:
                stall_mode_value = STALL_MODES[stall_mode_name]
                suffix = "" if stall_mode_name == "none" else f"_{stall_mode_name}"
                case_key = f"len_{length}{suffix}"
                capture_host = ARTIFACT_ROOT / "traces" / f"rtl_{schedule_name}_len_{length}{suffix}.captured"
                node_host = ARTIFACT_ROOT / "traces" / f"numeric_alignment_{schedule_name}_len_{length}{suffix}.jsonl"
                run_log_host = ARTIFACT_ROOT / "rtl_logs" / f"ml_m3_{schedule_name}_len_{length}{suffix}_{run_id}.log"
                run = _run_container_bash(
                    _run_script_with_options(
                        simv,
                        _container_path(vector_host),
                        _container_path(capture_host),
                        _container_path(node_host),
                        _container_path(run_log_host),
                        timeout_seconds=1800 if length <= 16 else 3600,
                        diagnostic=diagnostic,
                        stall_mode_value=stall_mode_value,
                    ),
                    timeout_seconds=3900,
                )
                parsed = _parse_log(run_log_host)
                case = {
                    "length": length,
                    "stall_mode": stall_mode_name,
                    "stall_mode_value": stall_mode_value,
                    "run_exit_code": run.returncode,
                    "run_stdout_tail": run.stdout[-4000:],
                    "run_log": str(run_log_host),
                    "capture": str(capture_host),
                    "capture_sha256": sha256_file(capture_host) if capture_host.exists() else None,
                    "node_trace": str(node_host),
                    "node_trace_sha256": sha256_file(node_host) if node_host.exists() else None,
                    **parsed,
                    "result": "PASS" if run.returncode == 0 and parsed["pass"] else ("DIAGNOSTIC" if diagnostic and run.returncode == 0 else "FAIL"),
                }
                schedule_result["cases"][case_key] = case
        schedule_result["result"] = "PASS" if all(case["result"] == "PASS" for case in schedule_result["cases"].values()) else "FAIL"
        results["schedules"][schedule_name] = schedule_result
    results["overall_result"] = "PASS" if all(row.get("result") == "PASS" for row in results["schedules"].values()) else "FAIL"
    write_json(ARTIFACT_ROOT / "comparisons" / "rtl_results.json", results)
    _write_rtl_reports(results)
    return results


def _write_rtl_reports(results: dict[str, Any]) -> None:
    reports = Path("reports/ml_m3")
    reports.mkdir(parents=True, exist_ok=True)
    smoke_lines = ["# ML-M3 RTL Smoke Results", ""]
    inc_lines = ["# ML-M3 Incremental KV Results", ""]
    ab_lines = ["# ML-M3 H8/H9 Real-Weight Comparison", ""]
    node_lines = ["# ML-M3 Node Comparison", ""]
    node_lines.append("The production RTL was not modified. Directly observed real-RTL boundaries are top output, top done, and internal ready/done handshakes exposed through read-only hierarchical references in the M3 testbench.")
    node_lines.extend(["", "| Schedule | Length | Stall | Result | Boundary line |", "|---|---:|---|---|---|"])
    ab_lines.extend(["| Case | H8 staged result | H9 interleaved result | H8 capture SHA | H9 capture SHA | Output identical |", "|---|---|---|---|---|---|"])
    for schedule_name, schedule in results["schedules"].items():
        for key, case in schedule.get("cases", {}).items():
            target = smoke_lines if case["length"] == 1 else inc_lines
            target.append(
                f"- {schedule_name} {key}: {case['result']} stall={case.get('stall_mode')} log=`{case['run_log']}` capture_sha={case['capture_sha256']}"
            )
            node_lines.append(
                f"| {schedule_name} | {case['length']} | {case.get('stall_mode')} | {case['result']} | `{case.get('boundary_line')}` |"
            )
    staged_cases = results["schedules"].get("staged", {}).get("cases", {})
    interleaved_cases = results["schedules"].get("interleaved", {}).get("cases", {})
    for key in sorted(set(staged_cases) | set(interleaved_cases)):
        staged = staged_cases.get(key, {})
        inter = interleaved_cases.get(key, {})
        same = staged.get("capture_sha256") is not None and staged.get("capture_sha256") == inter.get("capture_sha256")
        ab_lines.append(
            f"| {key} | {staged.get('result', 'NOT_RUN')} | "
            f"{inter.get('result', 'NOT_RUN')} | `{staged.get('capture_sha256')}` | `{inter.get('capture_sha256')}` | {same} |"
        )
    for path, lines in {
        "rtl_smoke_results.md": smoke_lines,
        "incremental_kv_results.md": inc_lines,
        "h8_h9_real_weight_comparison.md": ab_lines,
        "node_comparison.md": node_lines,
    }.items():
        lines.append("")
        (reports / path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", action="append", type=int, help="Length to run; may be repeated.")
    parser.add_argument("--schedule", choices=sorted(SCHEDULES), action="append", help="Schedule to run; may be repeated.")
    parser.add_argument("--run-id")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--stall-mode", choices=sorted(STALL_MODES), action="append", help="Ready/valid stall mode; may be repeated.")
    args = parser.parse_args()
    lengths = args.length or VECTOR_LENGTHS_REQUIRED
    schedules = args.schedule or ["staged", "interleaved"]
    print(json.dumps(run_vcs(lengths, schedules, run_id=args.run_id, diagnostic=args.diagnostic, stall_modes=args.stall_mode), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
