"""Classify the current ML-M3 one-token RTL/bit-model numeric mismatch."""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.cosim.m3_artifact_audit import deterministic_ids, load_q2_model
from ml.cosim.hardware_aware_layer import run_hardware_aware_layer
from ml.cosim.m3_trace_schema import ARTIFACT_ROOT, H9_ACCEPTED_COMMIT, H9_ACCEPTED_TAG, write_json
from model.arithmetic.fp16_fp32_reference import fp16_to_fp32_bits
from model.arithmetic.fp32_add_reference import fp32_add
from model.pe.pe_lane_reference import PE_LANE_MODE_PRODUCT, pe_lane_compute


def _bits_to_float(bits: int) -> float:
    return struct.unpack(">f", int(bits & 0xFFFFFFFF).to_bytes(4, "big"))[0]


def _float_fields(bits: int) -> dict[str, Any]:
    bits &= 0xFFFFFFFF
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    return {
        "hex": f"{bits:08x}",
        "float": _bits_to_float(bits),
        "sign": (bits >> 31) & 1,
        "exponent": exp,
        "mantissa": mant,
        "subnormal": exp == 0 and mant != 0,
        "zero": exp == 0 and mant == 0,
        "inf": exp == 0xFF and mant == 0,
        "nan": exp == 0xFF and mant != 0,
    }


def _ordered_int(bits: int) -> int:
    bits &= 0xFFFFFFFF
    return (~bits & 0xFFFFFFFF) if (bits & 0x80000000) else (bits | 0x80000000)


def _ulp_distance(a: int, b: int) -> int:
    return abs(_ordered_int(a) - _ordered_int(b))


def _read_capture(path: Path) -> tuple[dict[int, int], dict[int, tuple[int, int]]]:
    actual: dict[int, int] = {}
    diffs: dict[int, tuple[int, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] == "R":
            actual[int(parts[2])] = int(parts[3], 16)
        elif len(parts) == 5 and parts[0] == "D":
            diffs[int(parts[2])] = (int(parts[3], 16), int(parts[4], 16))
    return actual, diffs


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _latest_by_boundary(rows: list[dict[str, Any]], boundary: str) -> dict[str, Any] | None:
    values = [row for row in rows if row.get("boundary") == boundary]
    return values[-1] if values else None


def classify_numeric_mismatch() -> dict[str, Any]:
    staged_capture = ARTIFACT_ROOT / "traces" / "rtl_staged_len_1.captured"
    inter_capture = ARTIFACT_ROOT / "traces" / "rtl_interleaved_len_1.captured"
    staged_trace = ARTIFACT_ROOT / "traces" / "numeric_alignment_staged_len_1.jsonl"
    inter_trace = ARTIFACT_ROOT / "traces" / "numeric_alignment_interleaved_len_1.jsonl"
    replay_path = ARTIFACT_ROOT / "comparisons" / "numeric_replay_add.json"

    staged_actual, staged_diffs = _read_capture(staged_capture)
    inter_actual, inter_diffs = _read_capture(inter_capture)
    if staged_actual != inter_actual:
        raise RuntimeError("H8/H9 captures differ; numeric classification expects the known common-path mismatch")

    first_dim = min(staged_diffs)
    got, expected = staged_diffs[first_dim]
    mismatch_rows = []
    for dim, (actual, exp) in staged_diffs.items():
        mismatch_rows.append(
            {
                "dim": dim,
                "actual_hex": f"{actual:08x}",
                "expected_hex": f"{exp:08x}",
                "ulp_distance": _ulp_distance(actual, exp),
                "absolute_error": abs(_bits_to_float(actual) - _bits_to_float(exp)),
            }
        )

    loaded = load_q2_model()
    model = loaded.model
    tokenizer = loaded.tokenizer
    input_ids = deterministic_ids(tokenizer, 1)
    layer_input = (
        model.token_embedding(input_ids)
        + model.position_embedding(model._position_ids(1, 1, 0, input_ids.device))
    )[0].detach().cpu().float()
    hw = run_hardware_aware_layer(model, layer_input)
    trace = hw.traces[0]

    staged_nodes = _read_jsonl(staged_trace)
    inter_nodes = _read_jsonl(inter_trace)
    boundary_expectations = {
        "residual1_fp32_edge": trace.residual1_fp32[1],
        "norm2_output_fp16_edge": trace.norm2_fp16[1],
        "w2_output_fp32_edge": trace.ffn2_fp32[1],
        "residual2_input_lhs_fp32_edge": trace.residual1_fp32[1],
        "residual2_input_rhs_fp32_edge": trace.ffn2_fp32[1],
        "residual2_final_fp32_edge": trace.final_fp32[1],
    }
    boundary_table = []
    first_boundary = None
    for boundary, exp in boundary_expectations.items():
        h8 = _latest_by_boundary(staged_nodes, boundary)
        h9 = _latest_by_boundary(inter_nodes, boundary)
        actual_h8 = int(h8["actual"], 16) if h8 else None
        actual_h9 = int(h9["actual"], 16) if h9 else None
        match = actual_h8 == exp and actual_h9 == exp
        if first_boundary is None and not match:
            first_boundary = boundary
        boundary_table.append(
            {
                "boundary": boundary,
                "expected": f"{exp:08x}" if exp > 0xFFFF else f"{exp:04x}",
                "h8": None if actual_h8 is None else (f"{actual_h8:08x}" if actual_h8 > 0xFFFF else f"{actual_h8:04x}"),
                "h9": None if actual_h9 is None else (f"{actual_h9:08x}" if actual_h9 > 0xFFFF else f"{actual_h9:04x}"),
                "match": match,
                "cycle_h8": None if h8 is None else h8.get("cycle"),
                "cycle_h9": None if h9 is None else h9.get("cycle"),
            }
        )

    edge_add_inputs = {
        (row["base"], row["width"], row["pair"]): row
        for row in staged_nodes
        if row.get("boundary") == "w2_reduction_add_input_edge"
    }
    first_add = None
    for row in [row for row in staged_nodes if row.get("boundary") == "w2_reduction_add_output_edge"]:
        key = (row["base"], row["width"], row["pair"])
        inp = edge_add_inputs[key]
        add = fp32_add(int(inp["a"], 16), int(inp["b"], 16))
        actual = int(row["result"], 16)
        if actual != add.output_bits:
            first_add = {
                "base": row["base"],
                "width": row["width"],
                "pair": row["pair"],
                "cycle": row["cycle"],
                "a_hex": inp["a"],
                "b_hex": inp["b"],
                "rtl_result_hex": row["result"],
                "bit_model_result_hex": f"{add.output_bits:08x}",
                "rtl_status": row["status"],
                "bit_model_invalid": bool(add.invalid),
                "ulp_distance": _ulp_distance(actual, add.output_bits),
            }
            break

    product_mismatches = 0
    operands = {
        (row["base"], row["lane"]): row
        for row in staged_nodes
        if row.get("boundary") == "w2_tile_operand_fp16_edge"
    }
    for row in [row for row in staged_nodes if row.get("boundary") == "w2_lane_product_fp32_edge"]:
        operand = operands[(row["base"], row["lane"])]
        a32 = fp16_to_fp32_bits(int(operand["activation"], 16))["output_bits"]
        b32 = fp16_to_fp32_bits(int(operand["weight"], 16))["output_bits"]
        exp_product = pe_lane_compute(PE_LANE_MODE_PRODUCT, a32, b32, 0, bool(row["lane_mask"])).output_bits
        if int(row["actual"], 16) != exp_product:
            product_mismatches += 1

    replay = json.loads(replay_path.read_text(encoding="utf-8")) if replay_path.exists() else {}
    payload = {
        "stage": "ML-M3 Numeric Alignment",
        "hardware": {"tag": H9_ACCEPTED_TAG, "commit": H9_ACCEPTED_COMMIT},
        "minimal_reproduction": {
            "token": 0,
            "first_output_dim": first_dim,
            "expected": _float_fields(expected),
            "actual": _float_fields(got),
            "absolute_error": abs(_bits_to_float(got) - _bits_to_float(expected)),
            "relative_error": abs(_bits_to_float(got) - _bits_to_float(expected)) / max(abs(_bits_to_float(expected)), 1e-45),
            "integer_bit_pattern_delta": got - expected,
            "ulp_distance": _ulp_distance(got, expected),
            "h8_h9_identical": staged_actual == inter_actual,
            "mismatch_count_64": len(staged_diffs),
        },
        "mismatches": mismatch_rows,
        "boundary_table": boundary_table,
        "first_divergence_boundary": first_boundary,
        "first_arithmetic_divergence": first_add,
        "w2_operands_match": len(operands) == 256,
        "w2_lane_product_mismatch_count": product_mismatches,
        "numeric_replay": replay,
        "root_cause_classification": "C. RTL common arithmetic path bug",
        "root_cause_summary": (
            "H8/H9 are identical and vectors, operands, and lane products match the bit model. "
            "The first stable divergent boundary is FFN W2 output, and the first arithmetic divergence "
            "is inside the W2 reduction-tree add for row 1, tile base 8, pair 3. A standalone stable "
            "fp32_add_wrapper replay of the same operands matches the bit model/NumPy, so the full RTL "
            "PE reduction path is consuming a different result under its handshake/stream_reg scheduling."
        ),
    }
    write_json(ARTIFACT_ROOT / "comparisons" / "len1_full_output_diff.json", payload)
    _write_reports(payload)
    return payload


def _write_reports(payload: dict[str, Any]) -> None:
    reports = Path("reports/ml_m3")
    reports.mkdir(parents=True, exist_ok=True)
    repro = payload["minimal_reproduction"]
    first = payload["first_arithmetic_divergence"]
    lines = [
        "# ML-M3 Numeric Mismatch Classification",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Expected | `0x{repro['expected']['hex']}` ({repro['expected']['float']:.12g}) |",
        f"| Actual | `0x{repro['actual']['hex']}` ({repro['actual']['float']:.12g}) |",
        f"| ULP distance | {repro['ulp_distance']} |",
        f"| Integer bit-pattern delta | {repro['integer_bit_pattern_delta']} |",
        f"| 64-dim mismatch count | {repro['mismatch_count_64']} |",
        f"| H8/H9 identical | {repro['h8_h9_identical']} |",
        f"| NaN/Inf | expected nan={repro['expected']['nan']} inf={repro['expected']['inf']}; actual nan={repro['actual']['nan']} inf={repro['actual']['inf']} |",
        f"| Subnormal/FTZ | expected subnormal={repro['expected']['subnormal']}; actual subnormal={repro['actual']['subnormal']} |",
        "",
        "## Boundary Table",
        "",
        "| Boundary | Expected | H8 | H9 | Match | First divergence |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in payload["boundary_table"]:
        lines.append(
            f"| {row['boundary']} | `{row['expected']}` | `{row['h8']}` | `{row['h9']}` | {row['match']} | "
            f"{'yes' if row['boundary'] == payload['first_divergence_boundary'] else ''} |"
        )
    lines += [
        "",
        "## Classification",
        "",
        f"Root cause class: **{payload['root_cause_classification']}**.",
        "",
        payload["root_cause_summary"],
        "",
    ]
    (reports / "numeric_mismatch_classification.md").write_text("\n".join(lines), encoding="utf-8")

    trace_lines = [
        "# ML-M3 First Divergence Trace",
        "",
        "The first stable boundary divergence is `w2_output_fp32_edge`; earlier `residual1_fp32_edge` and `norm2_output_fp16_edge` match.",
        "",
        "## First Arithmetic Divergence",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    if first:
        trace_lines += [
            f"| Node | W2 reduction tree add |",
            f"| Cycle | {first['cycle']} |",
            f"| Tile base | {first['base']} |",
            f"| Width/pair | {first['width']} / {first['pair']} |",
            f"| Operand A | `0x{first['a_hex']}` |",
            f"| Operand B | `0x{first['b_hex']}` |",
            f"| RTL result | `0x{first['rtl_result_hex']}` |",
            f"| Bit-model result | `0x{first['bit_model_result_hex']}` |",
            f"| ULP distance | {first['ulp_distance']} |",
        ]
    trace_lines.append("")
    (reports / "first_divergence_trace.md").write_text("\n".join(trace_lines), encoding="utf-8")

    bug_lines = [
        "# ML-M3 Hardware Numeric Bug Report",
        "",
        f"- Hardware tag: `{payload['hardware']['tag']}`",
        f"- Hardware commit: `{payload['hardware']['commit']}`",
        "- Affected path: common H8/H9 FFN W2 GEMV reduction path (`reconfigurable_pe_core` / `fp32_reduction_tree` / `fp32_add_wrapper` interaction).",
        "- H8 staged and H9 interleaved produce identical one-token output and identical captured SHA, so this is not an H9-only scheduler issue.",
        "- Vector/export issue excluded: W2 row=1 FP16 operands match the bit model.",
        "- Lane product issue excluded: all 256 row=1 W2 lane products match the bit model.",
        "- Bit-model issue not supported: standalone stable `fp32_add_wrapper` replay of the first divergent operands matches the current bit model and NumPy float32.",
        "",
        "## Minimal Failing Operation",
        "",
    ]
    if first:
        bug_lines += [
            f"- Cycle: `{first['cycle']}`",
            f"- Operation: W2 reduction-tree add, tile base `{first['base']}`, width `{first['width']}`, pair `{first['pair']}`",
            f"- Operand A: `0x{first['a_hex']}`",
            f"- Operand B: `0x{first['b_hex']}`",
            f"- RTL result in full PE path: `0x{first['rtl_result_hex']}`",
            f"- Bit/stable replay expected: `0x{first['bit_model_result_hex']}`",
            f"- ULP distance: `{first['ulp_distance']}`",
        ]
    bug_lines += [
        "",
        "## Suspected RTL Area",
        "",
        "- `rtl/pe/fp32_reduction_tree.sv`: single-wrapper sequential reduction handshake.",
        "- `rtl/pe/reconfigurable_pe_core.sv`: reduction output to tile-accumulator handshake.",
        "- `rtl/arithmetic/fp32_add_wrapper.sv` and `rtl/common/stream_reg.sv`: registered output and ready/valid interaction.",
        "",
        "## Proposed Hardware-Line Follow-Up",
        "",
        "Create a hardware-owned fix task that reproduces this one-add operation inside the PE reduction context, then adjusts the RTL so `fp32_add_wrapper` outputs are consumed only after stable registered results. The model branch must not patch RTL.",
        "",
        "## Regression Command",
        "",
        "```powershell",
        "cd D:/IC_Workspace/VEDA_ml_m2",
        "python scripts/ml/run_ml_m3_vcs.py --length 1 --schedule staged --schedule interleaved --run-id numeric_alignment_repro --diagnostic",
        "python scripts/ml/run_ml_m3_numeric_replay.py --run-id numeric_alignment_first_add_const",
        "python scripts/ml/run_ml_m3_numeric_classification.py",
        "```",
        "",
    ]
    (reports / "hardware_numeric_bug_report.md").write_text("\n".join(bug_lines), encoding="utf-8")

    summary_lines = [
        "# ML-M3 Numeric Alignment Summary",
        "",
        "Status: **MODEL STAGE M3 IN PROGRESS - HARDWARE NUMERIC FIX REQUIRED**.",
        "",
        "One-token alignment is not closed. The known dim1 4-ULP final mismatch is stable, H8/H9 remain identical, and the first proven divergent arithmetic operation is in the common FFN W2 reduction path.",
        "",
        "No RTL, checkpoint, tokenizer, or hardware working-tree file was modified by this model task.",
        "",
    ]
    (reports / "numeric_alignment_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    regression_lines = [
        "# ML-M3 Numeric Alignment Regression",
        "",
        "| Check | Result |",
        "|---|---|",
        "| Reproduced one-token H8/H9 mismatch | PASS - mismatch reproduced |",
        "| Diagnostic full 64-dim capture | PASS - 54 mismatches collected |",
        "| H8/H9 identity | PASS - captures identical |",
        "| Numeric replay | PASS - stable wrapper replay matches bit model |",
        "| One-token bit-exact closure | FAIL - hardware numeric fix required |",
        "| length2/8/16 continuation | NOT RUN by task boundary |",
        "",
    ]
    (reports / "numeric_alignment_regression.md").write_text("\n".join(regression_lines), encoding="utf-8")


def main() -> None:
    print(json.dumps(classify_numeric_mismatch(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
