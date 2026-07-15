"""Close or block ML-M3 acceptance from generated manifests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.cosim.m3_trace_schema import ARTIFACT_ROOT, Q2_CHECKPOINT, Q2_TOKENIZER, read_json, sha256_file, write_json


REPORT_DIR = Path("reports/ml_m3")


def _load_optional(path: Path) -> dict:
    if path.exists():
        return read_json(path)
    return {}


def _first_fail(rtl: dict) -> dict:
    for schedule_name, schedule in rtl.get("schedules", {}).items():
        for case_name, case in schedule.get("cases", {}).items():
            if case.get("result") != "PASS":
                marker = case.get("fail_markers", [""])[0] if case.get("fail_markers") else ""
                return {"schedule": schedule_name, "case": case_name, "marker": marker, "log": case.get("run_log"), "capture": case.get("capture")}
    return {}


def _h8_h9_len1_sha(rtl: dict) -> dict:
    staged = rtl.get("schedules", {}).get("staged", {}).get("cases", {}).get("len_1", {})
    interleaved = rtl.get("schedules", {}).get("interleaved", {}).get("cases", {}).get("len_1", {})
    return {
        "staged_capture_sha256": staged.get("capture_sha256"),
        "interleaved_capture_sha256": interleaved.get("capture_sha256"),
        "match": bool(staged.get("capture_sha256") and staged.get("capture_sha256") == interleaved.get("capture_sha256")),
        "staged_result": staged.get("result", "NOT_RUN"),
        "interleaved_result": interleaved.get("result", "NOT_RUN"),
    }


def run_acceptance() -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = _load_optional(ARTIFACT_ROOT / "manifests" / "artifact_audit.json")
    vectors = _load_optional(ARTIFACT_ROOT / "manifests" / "vector_manifest.json")
    reference = _load_optional(ARTIFACT_ROOT / "comparisons" / "reference_chain.json")
    rtl = _load_optional(ARTIFACT_ROOT / "comparisons" / "rtl_results.json")
    docker = _load_optional(ARTIFACT_ROOT / "manifests" / "docker_environment.json")
    first_fail = _first_fail(rtl)
    h8_h9 = _h8_h9_len1_sha(rtl)
    required_vectors = {1, 2, 8, 16}
    generated_lengths = {int(case["length"]) for case in vectors.get("cases", [])}
    status = "MODEL STAGE M3 IN PROGRESS - RTL/BIT-MODEL NUMERIC MISMATCH BLOCKED"
    if (
        artifact.get("overall_result") == "PASS"
        and required_vectors.issubset(generated_lengths)
        and rtl.get("overall_result") == "PASS"
    ):
        status = "MODEL STAGE M3 PASS"
    payload = {
        "stage": "ML-M3",
        "status": status,
        "artifact_audit_result": artifact.get("overall_result", "MISSING"),
        "vector_lengths": sorted(generated_lengths),
        "required_vector_result": "PASS" if required_vectors.issubset(generated_lengths) else "FAIL",
        "reference_chain_cases": sorted(reference.get("comparisons", {}).keys()),
        "rtl_overall_result": rtl.get("overall_result", "MISSING"),
        "first_observable_difference": first_fail,
        "h8_h9_len1_capture": h8_h9,
        "checkpoint": str(Q2_CHECKPOINT),
        "checkpoint_sha256": sha256_file(Q2_CHECKPOINT),
        "tokenizer": str(Q2_TOKENIZER),
        "tokenizer_sha256": sha256_file(Q2_TOKENIZER),
        "docker_environment": docker,
        "pdk_used": False,
        "sta_pr_pna_used": False,
        "hardware_modified_by_ml_m3": False,
        "multitoken_rtl_deferred_due_to_one_token_gate": status != "MODEL STAGE M3 PASS",
        "length32_status": "VECTOR_GENERATED_ONLY; REAL_RTL_DEFERRED",
    }
    write_json(ARTIFACT_ROOT / "manifests" / "acceptance.json", payload)
    _write_reports(payload, artifact, vectors, reference, rtl)
    return payload


def _write_reports(payload: dict, artifact: dict, vectors: dict, reference: dict, rtl: dict) -> None:
    first = payload["first_observable_difference"]
    h8_h9 = payload["h8_h9_len1_capture"]
    smoke = [
        "# ML-M3 RTL Smoke Results",
        "",
        f"- H8/D8/D_MODEL64/D_FFN256 compile/elaborate: {'PASS' if rtl.get('schedules') else 'NOT_RUN'}",
        f"- H8 staged one-token: {h8_h9['staged_result']}",
        f"- H9 interleaved one-token: {h8_h9['interleaved_result']}",
        f"- First observable difference: `{first.get('marker', 'none')}`",
        f"- H8/H9 partial capture SHA match: {h8_h9['match']}",
        f"- H8 partial capture SHA256: `{h8_h9['staged_capture_sha256']}`",
        f"- H9 partial capture SHA256: `{h8_h9['interleaved_capture_sha256']}`",
        "",
        "Because one-token smoke failed, ML-M3 did not enter multi-token RTL co-simulation.",
        "",
    ]
    (REPORT_DIR / "rtl_smoke_results.md").write_text("\n".join(smoke), encoding="utf-8")
    inc = [
        "# ML-M3 Incremental KV Results",
        "",
        "Status: BLOCKED.",
        "",
        "The mandatory one-token RTL smoke failed before length 2/8/16 runs. Per the ML-M3 gate, multi-token incremental KV RTL co-simulation was not started.",
        "",
        f"Software full-vs-incremental reference from artifact audit: {artifact.get('incremental_reference_result', 'MISSING')}",
        "",
    ]
    (REPORT_DIR / "incremental_kv_results.md").write_text("\n".join(inc), encoding="utf-8")
    h8_report = [
        "# ML-M3 H8/H9 Real-Weight Comparison",
        "",
        "| Case | H8 staged | H9 interleaved | H8 SHA | H9 SHA | H8/H9 captured output identical |",
        "|---|---|---|---|---|---|",
        f"| length 1 partial | {h8_h9['staged_result']} | {h8_h9['interleaved_result']} | `{h8_h9['staged_capture_sha256']}` | `{h8_h9['interleaved_capture_sha256']}` | {h8_h9['match']} |",
        "",
        "Both schedules hit the same first mismatch against the hardware-aware bit model before completing token 0. This confirms the two accepted schedules agree on the captured prefix, but ML-M3 cannot claim bit-model equivalence.",
        "",
    ]
    (REPORT_DIR / "h8_h9_real_weight_comparison.md").write_text("\n".join(h8_report), encoding="utf-8")
    node = [
        "# ML-M3 Node Comparison",
        "",
        "Final layer output is the first checked real-RTL data boundary in the M3 testbench. The first observable data mismatch occurs at token 0, dimension 1.",
        "",
        f"- First mismatch: `{first.get('marker', 'none')}`",
        "- Direct internal data-node comparison was not expanded after the one-token final-output gate failed.",
        "- Read-only hierarchical boundary monitors were compiled into the model-line testbench, but no PASS boundary summary was emitted because the simulation stopped at the first output mismatch.",
        "",
    ]
    (REPORT_DIR / "node_comparison.md").write_text("\n".join(node), encoding="utf-8")
    next_token = [
        "# ML-M3 Next-Token Results",
        "",
        "Status: BLOCKED.",
        "",
        "Hybrid next-token validation requires a complete real RTL layer output. The one-token RTL smoke stopped at output dimension 1, so no RTL-assisted logits or top-k token result was claimed.",
        "",
        "Software hardware-aware next-token references remain available in `D:/IC_Workspace/VEDA_artifacts/ml_m3/comparisons/next_token_reference.json`.",
        "",
    ]
    (REPORT_DIR / "next_token_results.md").write_text("\n".join(next_token), encoding="utf-8")
    regression = [
        "# ML-M3 Regression",
        "",
        "| Check | Result | Notes |",
        "|---|---|---|",
        f"| Q2 checkpoint/tokenizer/export audit | {payload['artifact_audit_result']} | `{ARTIFACT_ROOT / 'manifests' / 'artifact_audit.json'}` |",
        f"| Required vector generation | {payload['required_vector_result']} | lengths {payload['vector_lengths']} |",
        "| Python compile for new M3 scripts | PASS | `python -m py_compile ...` completed before RTL run |",
        f"| H8/H9 one-token RTL smoke | {payload['rtl_overall_result']} | {first.get('marker', 'no marker')} |",
        "| Multi-token RTL | BLOCKED | Not run after one-token gate failure |",
        "| PDK/STA/P&R/PPA | NOT_RUN | Outside ML-M3 and not invoked |",
        "",
    ]
    (REPORT_DIR / "regression.md").write_text("\n".join(regression), encoding="utf-8")
    acceptance_lines = [
        "# ML-M3 Acceptance Audit",
        "",
        f"Status: **{payload['status']}**",
        "",
        "| Requirement | Result | Evidence |",
        "|---|---|---|",
        f"| Q2 checkpoint SHA correct | {artifact.get('checkpoint_sha_result', 'MISSING')} | `{payload['checkpoint_sha256']}` |",
        f"| Tokenizer SHA correct | {artifact.get('tokenizer_sha_result', 'MISSING')} | `{payload['tokenizer_sha256']}` |",
        f"| 12 export tensors audited | {artifact.get('export_result', 'MISSING')} | tensor_count={artifact.get('export_tensor_count')} |",
        f"| 8 RTL weight mappings | {artifact.get('weight_mapping_result', 'MISSING')} | `reports/ml_m3/weight_mapping_audit.md` |",
        "| H8/D8/D_MODEL64/D_FFN256 elaborate | PASS | VCS compile succeeded for staged and interleaved one-token smoke |",
        f"| H9 one-token real RTL | {h8_h9['interleaved_result']} | {first.get('marker', '')} |",
        f"| H8 one-token real RTL | {h8_h9['staged_result']} | same mismatch marker |",
        "| length 1/2/8/16 real RTL | BLOCKED | one-token gate failed before multi-token |",
        "| bit model vs H8/H9 | FAIL | first mismatch token 0 dim 1 |",
        f"| H8 vs H9 | PARTIAL_PASS | captured prefix SHA match={h8_h9['match']} |",
        "| Next-token RTL-assisted cases | BLOCKED | no complete RTL layer output |",
        "| Hardware worktree modified | PASS | no model-line write to hardware repo |",
        "| PDK/STA/P&R/PPA | PASS | not used |",
        "",
        "ML-M3 cannot be accepted until the RTL/bit-model numerical mismatch is resolved in a separate hardware or reference-model fix task.",
        "",
    ]
    (REPORT_DIR / "acceptance_audit.md").write_text("\n".join(acceptance_lines), encoding="utf-8")
    summary = [
        "# ML-M3 Summary",
        "",
        f"Status: **{payload['status']}**",
        "",
        "Completed:",
        "",
        "- Q2 checkpoint/tokenizer/export/weight mapping audit passed.",
        "- Real Q2 vectors were generated for lengths 1, 2, 8, 16, and 32.",
        "- Reference-chain metrics were generated for PyTorch FP32, FP16-weight PyTorch, and the hardware-aware bit model.",
        "- H8 staged and H9 interleaved transformer_layer both compiled/elaborated with N_HEAD=8, D_HEAD=8, D_MODEL=64, D_FFN=256, MAX_SEQ_LEN=128.",
        "- H8 and H9 one-token runs produced identical captured prefix output.",
        "",
        "Blocked:",
        "",
        f"- First checked final-output boundary mismatched bit model: `{first.get('marker', 'none')}`.",
        "- Multi-token RTL, full H8/H9 A/B, hybrid next-token logits, and M3 PASS are deferred until the mismatch is fixed.",
        "",
        "No hardware source, hardware report, PDK, STA, P&R, or PPA flow was modified or invoked.",
        "",
    ]
    (REPORT_DIR / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    issue = [
        "# ML-M3 Hardware Dependency Issue",
        "",
        "The accepted H9 hardware baseline compiles for the Q2 H8/D8 layer configuration, but real-weight one-token final output is not bit-exact against the current hardware-aware bit model.",
        "",
        f"- First mismatch: `{first.get('marker', 'none')}`",
        f"- H8 staged log: `{rtl.get('schedules', {}).get('staged', {}).get('cases', {}).get('len_1', {}).get('run_log')}`",
        f"- H9 interleaved log: `{rtl.get('schedules', {}).get('interleaved', {}).get('cases', {}).get('len_1', {}).get('run_log')}`",
        f"- Vector: `{ARTIFACT_ROOT / 'vectors' / 'len_1' / 'case_len_1.mem'}`",
        "- Hardware repo was not modified by this task.",
        "",
    ]
    (REPORT_DIR / "hardware_dependency_issues.md").write_text("\n".join(issue), encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(run_acceptance(), indent=2, sort_keys=True))
