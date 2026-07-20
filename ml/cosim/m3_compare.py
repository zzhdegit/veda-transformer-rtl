"""Compare ML-M3 RTL captures and run hybrid next-token validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ml.cosim.fp16_policy import fp32_bits_to_float, tensor_to_fp16_bits
from ml.cosim.hardware_aware_layer import run_hardware_aware_layer
from ml.cosim.hardware_aware_model import run_hardware_aware_model
from ml.cosim.m3_artifact_audit import deterministic_ids, load_q2_model
from ml.cosim.m3_trace_schema import ARTIFACT_ROOT, NEXT_TOKEN_LENGTHS, RTL_WEIGHT_STATE, VECTOR_LENGTHS_REQUIRED, read_json, sha256_file, write_json
from ml.evaluation.evaluate_quantization import logits_agreement, tensor_error_metrics
from model.arithmetic.fp16_fp32_reference import fp16_to_fp32_bits
from model.arithmetic.fp32_add_reference import fp32_add
from model.pe.pe_lane_reference import PE_LANE_MODE_PRODUCT, pe_lane_compute


def read_capture(path: str | Path, length: int, d_model: int = 64) -> tuple[torch.Tensor, list[list[int]]]:
    bits = [[None for _ in range(d_model)] for _ in range(length)]
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 4 or parts[0] != "R":
            continue
        token = int(parts[1])
        dim = int(parts[2])
        value = int(parts[3], 16)
        if 0 <= token < length and 0 <= dim < d_model:
            bits[token][dim] = value
    missing = [(tok, dim) for tok in range(length) for dim in range(d_model) if bits[tok][dim] is None]
    if missing:
        raise RuntimeError(f"{path} missing {len(missing)} output values")
    int_bits = [[int(value) for value in row] for row in bits]
    floats = [[fp32_bits_to_float(value) for value in row] for row in int_bits]
    return torch.tensor(floats, dtype=torch.float32).unsqueeze(0), int_bits


def read_expected_vector_bits(path: str | Path, length: int, d_model: int = 64) -> tuple[torch.Tensor, list[list[int]]]:
    bits = [[None for _ in range(d_model)] for _ in range(length)]
    current_token = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "T" and len(parts) >= 2:
            current_token = int(parts[1], 0)
        elif parts[0] == "O" and len(parts) == 3:
            dim = int(parts[1], 16)
            value = int(parts[2], 16)
            if 0 <= current_token < length and 0 <= dim < d_model:
                bits[current_token][dim] = value
    missing = [(tok, dim) for tok in range(length) for dim in range(d_model) if bits[tok][dim] is None]
    if missing:
        raise RuntimeError(f"{path} missing {len(missing)} expected output values")
    int_bits = [[int(value) for value in row] for row in bits]
    floats = [[fp32_bits_to_float(value) for value in row] for row in int_bits]
    return torch.tensor(floats, dtype=torch.float32).unsqueeze(0), int_bits


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _latest_boundary(rows: list[dict[str, Any]], boundary: str) -> dict[str, Any] | None:
    hits = [row for row in rows if row.get("boundary") == boundary and int(row.get("token", 0)) == 0]
    return hits[-1] if hits else None


@torch.no_grad()
def _hybrid_logits(model, layer_output: torch.Tensor) -> torch.Tensor:
    return model.lm_head(model.final_norm(layer_output))


def _top_record(logits: torch.Tensor, eos_id: int) -> dict[str, Any]:
    last = logits[0, -1].detach().float()
    probs = torch.softmax(last, dim=-1)
    top_values, top_indices = torch.topk(probs, k=5)
    sorted_ids = torch.argsort(probs, descending=True)
    eos_rank = int((sorted_ids == eos_id).nonzero(as_tuple=False)[0].item() + 1)
    return {
        "top1": int(top_indices[0].item()),
        "top5": [int(idx) for idx in top_indices.tolist()],
        "top5_probabilities": [float(value) for value in top_values.tolist()],
        "eos_rank": eos_rank,
        "eos_probability": float(probs[eos_id].item()),
    }


def _add_node_result(summary: dict[str, Any], category: str, checks: int, mismatches: int, example: str = "") -> None:
    row = summary.setdefault(category, {"checks": 0, "mismatches": 0, "example": ""})
    row["checks"] += checks
    row["mismatches"] += mismatches
    if example and not row["example"]:
        row["example"] = example


def _compare_node_trace(path: str | Path, schedule: str, model, tokenizer) -> dict[str, Any]:
    rows = _read_jsonl(path)
    input_ids = deterministic_ids(tokenizer, 1)
    layer_input = (
        model.token_embedding(input_ids)
        + model.position_embedding(model._position_ids(1, 1, 0, input_ids.device))
    )[0].detach().cpu().float()
    trace = run_hardware_aware_layer(model, layer_input).traces[0]
    w2_bits = tensor_to_fp16_bits(model.state_dict()[RTL_WEIGHT_STATE["w2"]])
    summary: dict[str, Any] = {}

    scalar_boundaries = {
        "residual1_fp32_edge": (trace.residual1_fp32[1], 32),
        "norm2_output_fp16_edge": (trace.norm2_fp16[1], 16),
        "w2_output_fp32_edge": (trace.ffn2_fp32[1], 32),
        "residual2_input_lhs_fp32_edge": (trace.residual1_fp32[1], 32),
        "residual2_input_rhs_fp32_edge": (trace.ffn2_fp32[1], 32),
        "residual2_final_fp32_edge": (trace.final_fp32[1], 32),
    }
    for boundary, (expected, width) in scalar_boundaries.items():
        row = _latest_boundary(rows, boundary)
        if row is None:
            _add_node_result(summary, boundary, 1, 1, "missing")
            continue
        actual = int(row["actual"], 16)
        _add_node_result(summary, boundary, 1, 0 if actual == expected else 1, f"actual=0x{actual:0{width // 4}x} expected=0x{expected:0{width // 4}x}")

    operand_rows = [row for row in rows if row.get("boundary") == "w2_tile_operand_fp16_edge" and int(row.get("token", 0)) == 0]
    operand_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    operand_mismatches = 0
    for row in operand_rows:
        base = int(row["base"])
        lane = int(row["lane"])
        idx = base + lane
        operand_by_key[(base, lane)] = row
        expected_activation = int(trace.activation_fp16[idx])
        expected_weight = int(w2_bits[1][idx])
        actual_activation = int(row["activation"], 16)
        actual_weight = int(row["weight"], 16)
        if actual_activation != expected_activation or actual_weight != expected_weight:
            operand_mismatches += 1
    _add_node_result(summary, "w2_tile_operand_fp16_edge", len(operand_rows), operand_mismatches)

    product_rows = [row for row in rows if row.get("boundary") == "w2_lane_product_fp32_edge" and int(row.get("token", 0)) == 0]
    product_mismatches = 0
    for row in product_rows:
        operand = operand_by_key[(int(row["base"]), int(row["lane"]))]
        a32 = fp16_to_fp32_bits(int(operand["activation"], 16))["output_bits"]
        b32 = fp16_to_fp32_bits(int(operand["weight"], 16))["output_bits"]
        expected = pe_lane_compute(PE_LANE_MODE_PRODUCT, a32, b32, 0, bool(int(row["lane_mask"]))).output_bits
        if int(row["actual"], 16) != expected:
            product_mismatches += 1
    _add_node_result(summary, "w2_lane_product_fp32_edge", len(product_rows), product_mismatches)

    add_inputs = {
        (int(row["base"]), int(row["width"]), int(row["pair"])): row
        for row in rows
        if row.get("boundary") == "w2_reduction_add_input_edge" and int(row.get("token", 0)) == 0
    }
    add_outputs = [row for row in rows if row.get("boundary") == "w2_reduction_add_output_edge" and int(row.get("token", 0)) == 0]
    add_mismatches = 0
    for row in add_outputs:
        inp = add_inputs.get((int(row["base"]), int(row["width"]), int(row["pair"])))
        if inp is None:
            add_mismatches += 1
            continue
        expected = fp32_add(int(inp["a"], 16), int(inp["b"], 16)).output_bits
        if int(row["result"], 16) != expected:
            add_mismatches += 1
    _add_node_result(summary, "w2_reduction_add_output_edge", len(add_outputs), add_mismatches)

    return {
        "schedule": schedule,
        "node_trace": str(path),
        "categories": summary,
        "category_count": len(summary),
        "check_count": sum(row["checks"] for row in summary.values()),
        "mismatch_count": sum(row["mismatches"] for row in summary.values()),
        "result": "PASS" if summary and all(row["mismatches"] == 0 for row in summary.values()) else "FAIL",
    }


def _compare_nodes(rtl_results: dict[str, Any], model, tokenizer) -> dict[str, Any]:
    schedules = {}
    for schedule_name in ["staged", "interleaved"]:
        case = rtl_results.get("schedules", {}).get(schedule_name, {}).get("cases", {}).get("len_1", {})
        if case.get("result") != "PASS" or not case.get("node_trace"):
            schedules[schedule_name] = {"schedule": schedule_name, "result": "MISSING", "categories": {}, "category_count": 0, "check_count": 0, "mismatch_count": 0}
            continue
        schedules[schedule_name] = _compare_node_trace(case["node_trace"], schedule_name, model, tokenizer)
    return {
        "stage": "ML-M3",
        "schedules": schedules,
        "overall_result": "PASS" if schedules and all(row.get("result") == "PASS" for row in schedules.values()) else "FAIL",
    }


def compare_rtl_outputs() -> dict[str, Any]:
    loaded = load_q2_model()
    model = loaded.model
    tokenizer = loaded.tokenizer
    rtl_results_path = ARTIFACT_ROOT / "comparisons" / "rtl_results.json"
    if not rtl_results_path.exists():
        raise RuntimeError("rtl_results.json is missing; run scripts/ml/run_ml_m3_vcs.py first")
    rtl_results = read_json(rtl_results_path)
    comparisons: dict[str, Any] = {"stage": "ML-M3", "cases": {}, "next_token": {}, "h8_h9": {}}
    for length in VECTOR_LENGTHS_REQUIRED:
        staged = rtl_results["schedules"].get("staged", {}).get("cases", {}).get(f"len_{length}", {})
        inter = rtl_results["schedules"].get("interleaved", {}).get("cases", {}).get(f"len_{length}", {})
        if staged.get("result") != "PASS" or inter.get("result") != "PASS":
            comparisons["cases"][f"len_{length}"] = {"result": "NOT_PASS", "staged": staged.get("result"), "interleaved": inter.get("result")}
            continue
        staged_layer, staged_bits = read_capture(staged["capture"], length)
        inter_layer, inter_bits = read_capture(inter["capture"], length)
        vector_path = ARTIFACT_ROOT / "vectors" / f"len_{length}" / f"case_len_{length}.mem"
        expected_layer, expected_bits = read_expected_vector_bits(vector_path, length)
        input_ids = deterministic_ids(tokenizer, length)
        hw = run_hardware_aware_model(model, input_ids)
        h8_logits = _hybrid_logits(model, staged_layer)
        h9_logits = _hybrid_logits(model, inter_layer)
        bit_logits = hw["logits"]
        h8_bit_exact = staged_bits == expected_bits
        h9_bit_exact = inter_bits == expected_bits
        h8_h9_bit_exact = staged_bits == inter_bits
        case = {
            "result": "PASS" if h8_bit_exact and h9_bit_exact and h8_h9_bit_exact else "FAIL",
            "h8_vs_h9_capture_sha_match": staged.get("capture_sha256") == inter.get("capture_sha256"),
            "h8_vs_bit_model_bit_exact": h8_bit_exact,
            "h9_vs_bit_model_bit_exact": h9_bit_exact,
            "h8_vs_h9_bit_exact": h8_h9_bit_exact,
            "h8_vs_bit_model_layer": tensor_error_metrics(staged_layer, expected_layer),
            "h9_vs_bit_model_layer": tensor_error_metrics(inter_layer, expected_layer),
            "h8_vs_h9_layer": tensor_error_metrics(staged_layer, inter_layer),
            "h8_logits_vs_bit_model": {**tensor_error_metrics(h8_logits, bit_logits), **logits_agreement(bit_logits, h8_logits)},
            "h9_logits_vs_bit_model": {**tensor_error_metrics(h9_logits, bit_logits), **logits_agreement(bit_logits, h9_logits)},
            "h8_top": _top_record(h8_logits, tokenizer.eos_id),
            "h9_top": _top_record(h9_logits, tokenizer.eos_id),
            "bit_top": _top_record(bit_logits, tokenizer.eos_id),
            "staged_capture_sha256": sha256_file(staged["capture"]),
            "interleaved_capture_sha256": sha256_file(inter["capture"]),
        }
        if length in NEXT_TOKEN_LENGTHS:
            comparisons["next_token"][f"len_{length}"] = {
                "prompt_token_ids": [int(idx) for idx in input_ids[0].tolist()],
                "h8_top": case["h8_top"],
                "h9_top": case["h9_top"],
                "bit_top": case["bit_top"],
                "top1_pass": case["h8_top"]["top1"] == case["h9_top"]["top1"] == case["bit_top"]["top1"],
            }
        comparisons["cases"][f"len_{length}"] = case
        comparisons["h8_h9"][f"len_{length}"] = {
            "capture_sha_match": staged.get("capture_sha256") == inter.get("capture_sha256"),
            "bit_exact_layer_outputs": h8_h9_bit_exact,
        }
    len1_next = comparisons["next_token"].get("len_1", {})
    len2_case = comparisons["cases"].get("len_2", {})
    len2_ids = deterministic_ids(tokenizer, 2)[0].tolist()
    comparisons["continuous_two_step"] = {
        "prompt0_token_ids": [int(deterministic_ids(tokenizer, 1)[0, 0].item())],
        "step0_bit_top1": len1_next.get("bit_top", {}).get("top1"),
        "step0_h8_top1": len1_next.get("h8_top", {}).get("top1"),
        "step0_h9_top1": len1_next.get("h9_top", {}).get("top1"),
        "prompt1_token_ids": [int(value) for value in len2_ids],
        "prompt1_uses_step0_prediction": bool(len1_next and int(len2_ids[1]) == len1_next["bit_top"]["top1"]),
        "step1_bit_top1": len2_case.get("bit_top", {}).get("top1"),
        "step1_h8_top1": len2_case.get("h8_top", {}).get("top1"),
        "step1_h9_top1": len2_case.get("h9_top", {}).get("top1"),
    }
    two = comparisons["continuous_two_step"]
    two["result"] = "PASS" if (
        two["prompt1_uses_step0_prediction"]
        and two["step0_bit_top1"] == two["step0_h8_top1"] == two["step0_h9_top1"]
        and two["step1_bit_top1"] == two["step1_h8_top1"] == two["step1_h9_top1"]
    ) else "FAIL"
    comparisons["node_comparison"] = _compare_nodes(rtl_results, model, tokenizer)
    comparisons["overall_result"] = "PASS" if all(row.get("result") == "PASS" for row in comparisons["cases"].values()) else "FAIL"
    if comparisons["continuous_two_step"]["result"] != "PASS" or comparisons["node_comparison"]["overall_result"] != "PASS":
        comparisons["overall_result"] = "FAIL"
    write_json(ARTIFACT_ROOT / "comparisons" / "rtl_hybrid_comparison.json", comparisons)
    write_json(ARTIFACT_ROOT / "comparisons" / "node_comparison.json", comparisons["node_comparison"])
    _write_reports(comparisons)
    return comparisons


def _write_reports(payload: dict[str, Any]) -> None:
    reports = Path("reports/ml_m3")
    reports.mkdir(parents=True, exist_ok=True)
    next_lines = [
        "# ML-M3 Hybrid Next-Token Results",
        "",
        "| Case | Prompt token IDs | Bit top-1 | H8 top-1 | H9 top-1 | Top-1 pass |",
        "|---|---|---:|---:|---:|---|",
    ]
    for name, row in payload["next_token"].items():
        next_lines.append(
            f"| {name} | {row['prompt_token_ids']} | {row['bit_top']['top1']} | "
            f"{row['h8_top']['top1']} | {row['h9_top']['top1']} | {row['top1_pass']} |"
        )
    two = payload.get("continuous_two_step", {})
    next_lines.extend([
        "",
        "## Continuous Two-Step",
        "",
        "| Step | Prompt token IDs | Bit top-1 | H8 top-1 | H9 top-1 | Result |",
        "|---:|---|---:|---:|---:|---|",
        f"| 0 | {two.get('prompt0_token_ids')} | {two.get('step0_bit_top1')} | {two.get('step0_h8_top1')} | {two.get('step0_h9_top1')} | {'PASS' if two.get('step0_bit_top1') == two.get('step0_h8_top1') == two.get('step0_h9_top1') else 'FAIL'} |",
        f"| 1 | {two.get('prompt1_token_ids')} | {two.get('step1_bit_top1')} | {two.get('step1_h8_top1')} | {two.get('step1_h9_top1')} | {'PASS' if two.get('result') == 'PASS' else 'FAIL'} |",
    ])
    next_lines.append("")
    (reports / "next_token_results.md").write_text("\n".join(next_lines), encoding="utf-8")
    h8_lines = [
        "# ML-M3 H8/H9 Real-Weight Comparison",
        "",
        "| Case | H8 vs H9 bit-exact | Capture SHA match | H8 vs bit max abs | H9 vs bit max abs | Top-1 agreement |",
        "|---|---|---|---:|---:|---:|",
    ]
    for name, row in payload["cases"].items():
        if row.get("result") != "PASS":
            h8_lines.append(f"| {name} | false | false | n/a | n/a | 0 |")
            continue
        h8_lines.append(
            f"| {name} | {payload['h8_h9'][name]['bit_exact_layer_outputs']} | {payload['h8_h9'][name]['capture_sha_match']} | "
            f"{row['h8_vs_bit_model_layer']['max_abs_error']:.6g} | {row['h9_vs_bit_model_layer']['max_abs_error']:.6g} | "
            f"{row['h9_logits_vs_bit_model']['top1_agreement']:.3f} |"
        )
    h8_lines.append("")
    (reports / "h8_h9_real_weight_comparison.md").write_text("\n".join(h8_lines), encoding="utf-8")
    node_lines = [
        "# ML-M3 Node Comparison",
        "",
        "Read-only model-line testbench monitors compared real RTL internal nodes against the hardware-aware bit model for the repaired length1 no-stall run.",
        "",
        "| Schedule | Category | Checks | Mismatches | Result |",
        "|---|---|---:|---:|---|",
    ]
    for schedule, schedule_row in payload.get("node_comparison", {}).get("schedules", {}).items():
        for category, row in schedule_row.get("categories", {}).items():
            node_lines.append(
                f"| {schedule} | `{category}` | {row['checks']} | {row['mismatches']} | {'PASS' if row['mismatches'] == 0 else 'FAIL'} |"
            )
    node_lines.extend([
        "",
        f"Overall: **{payload.get('node_comparison', {}).get('overall_result', 'MISSING')}**",
        "",
    ])
    (reports / "node_comparison.md").write_text("\n".join(node_lines), encoding="utf-8")


def main() -> None:
    print(json.dumps(compare_rtl_outputs(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
