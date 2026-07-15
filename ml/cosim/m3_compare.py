"""Compare ML-M3 RTL captures and run hybrid next-token validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ml.cosim.fp16_policy import fp32_bits_to_float
from ml.cosim.hardware_aware_model import run_hardware_aware_model
from ml.cosim.m3_artifact_audit import deterministic_ids, load_q2_model
from ml.cosim.m3_trace_schema import ARTIFACT_ROOT, NEXT_TOKEN_LENGTHS, VECTOR_LENGTHS_REQUIRED, read_json, sha256_file, write_json
from ml.evaluation.evaluate_quantization import logits_agreement, tensor_error_metrics


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
        input_ids = deterministic_ids(tokenizer, length)
        hw = run_hardware_aware_model(model, input_ids)
        hw_layer = hw["layer_output"]
        h8_logits = _hybrid_logits(model, staged_layer)
        h9_logits = _hybrid_logits(model, inter_layer)
        bit_logits = hw["logits"]
        case = {
            "result": "PASS" if staged_bits == inter_bits else "FAIL",
            "h8_vs_h9_capture_sha_match": staged.get("capture_sha256") == inter.get("capture_sha256"),
            "h8_vs_bit_model_layer": tensor_error_metrics(staged_layer, hw_layer),
            "h9_vs_bit_model_layer": tensor_error_metrics(inter_layer, hw_layer),
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
            "bit_exact_layer_outputs": staged_bits == inter_bits,
        }
    comparisons["overall_result"] = "PASS" if all(row.get("result") == "PASS" for row in comparisons["cases"].values()) else "FAIL"
    write_json(ARTIFACT_ROOT / "comparisons" / "rtl_hybrid_comparison.json", comparisons)
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


def main() -> None:
    print(json.dumps(compare_rtl_outputs(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
