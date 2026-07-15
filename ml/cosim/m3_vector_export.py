"""Generate real Q2 input vectors for ML-M3 RTL co-simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import torch

from ml.cosim.fp16_policy import float_to_fp16_bits
from ml.cosim.hardware_aware_layer import run_hardware_aware_layer
from ml.cosim.hardware_aware_model import run_hardware_aware_model
from ml.cosim.m3_artifact_audit import deterministic_ids, load_q2_model
from ml.cosim.m3_reference_chain import compare_reference_chain
from ml.cosim.m3_trace_schema import (
    ARTIFACT_ROOT,
    NEXT_TOKEN_LENGTHS,
    RTL_WEIGHT_KINDS,
    RTL_WEIGHT_STATE,
    VECTOR_LENGTHS_EXTENDED,
    VECTOR_LENGTHS_REQUIRED,
    ensure_artifact_dirs,
    sha256_file,
    write_json,
)
from ml.cosim.fp16_policy import tensor_to_fp16_bits


def _bits_checksum(values: list[int], width: int) -> str:
    pack = "<H" if width == 16 else "<I"
    hasher = hashlib.sha256()
    for value in values:
        hasher.update(struct.pack(pack, int(value) & ((1 << width) - 1)))
    return hasher.hexdigest()


def _flatten(obj) -> list[int]:
    if isinstance(obj, (list, tuple)):
        out = []
        for item in obj:
            out.extend(_flatten(item))
        return out
    return [int(obj)]


def _token_top_info(logits: torch.Tensor, eos_id: int) -> dict[str, Any]:
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


def _write_weights(handle, state: dict[str, torch.Tensor]) -> dict[str, Any]:
    records = []
    for name in ["wq", "wk", "wv", "wo", "norm1_gamma", "norm2_gamma", "w1", "w2"]:
        bits = tensor_to_fp16_bits(state[RTL_WEIGHT_STATE[name]])
        flat = _flatten(bits)
        kind = RTL_WEIGHT_KINDS[name]
        if name in {"norm1_gamma", "norm2_gamma"}:
            for row, value in enumerate(flat):
                handle.write(f"W {kind:d} {row:x} 0 {value & 0xFFFF:04x}\n")
        else:
            rows = bits
            for row, row_values in enumerate(rows):
                for col, value in enumerate(row_values):
                    handle.write(f"W {kind:d} {row:x} {col:x} {int(value) & 0xFFFF:04x}\n")
        records.append(
            {
                "tensor": name,
                "kind": kind,
                "state_dict": RTL_WEIGHT_STATE[name],
                "element_count": len(flat),
                "fp16_bits_sha256": _bits_checksum(flat, 16),
            }
        )
    return {"records": records, "total_elements": sum(row["element_count"] for row in records)}


def _node_checksums(trace) -> dict[str, str]:
    nodes = {
        "layer_input_fp32": trace.input_fp32,
        "norm1_output_fp16": trace.norm1_fp16,
        "q_projection_fp32": trace.mha.qkv.q_fp32,
        "q_projection_fp16": trace.mha.qkv.q_fp16,
        "k_projection_fp32": trace.mha.qkv.k_fp32,
        "k_projection_fp16": trace.mha.qkv.k_fp16,
        "v_projection_fp32": trace.mha.qkv.v_fp32,
        "v_projection_fp16": trace.mha.qkv.v_fp16,
        "scaled_attention_score_fp32": [head.scaled_scores for head in trace.mha.attention.attention_traces],
        "softmax_probability_fp32": [head.probabilities for head in trace.mha.attention.attention_traces],
        "head_output_fp32": trace.mha.head_output_fp32,
        "concat_fp16": trace.mha.concat_fp16,
        "wo_output_fp32": trace.mha.final_output_fp32,
        "residual1_fp32": trace.residual1_fp32,
        "norm2_output_fp16": trace.norm2_fp16,
        "w1_output_fp32": trace.ffn1_fp32,
        "relu_activation_fp16": trace.activation_fp16,
        "w2_output_fp32": trace.ffn2_fp32,
        "residual2_final_fp32": trace.final_fp32,
    }
    checksums = {}
    for name, values in nodes.items():
        flat = _flatten(values)
        width = 16 if name.endswith("_fp16") or name in {"norm1_output_fp16", "norm2_output_fp16", "concat_fp16", "relu_activation_fp16"} else 32
        checksums[name] = _bits_checksum(flat, width)
    return checksums


@torch.no_grad()
def generate_vectors(lengths: list[int] | None = None) -> dict[str, Any]:
    ensure_artifact_dirs()
    loaded = load_q2_model()
    model = loaded.model
    tokenizer = loaded.tokenizer
    state = model.state_dict()
    lengths = lengths or [*VECTOR_LENGTHS_REQUIRED, *VECTOR_LENGTHS_EXTENDED]
    case_records = []
    reference_records = {}
    next_token_records = {}
    for length in lengths:
        input_ids = deterministic_ids(tokenizer, length)
        position_ids = list(range(length))
        token_emb = model.token_embedding(input_ids)
        pos = model._position_ids(1, length, 0, input_ids.device)
        pos_emb = model.position_embedding(pos)
        layer_input = (token_emb + pos_emb)[0].detach().cpu().float()
        hw_layer = run_hardware_aware_layer(model, layer_input)
        hw_model = run_hardware_aware_model(model, input_ids)
        case_dir = ARTIFACT_ROOT / "vectors" / f"len_{length}"
        case_dir.mkdir(parents=True, exist_ok=True)
        vector_path = case_dir / f"case_len_{length}.mem"
        hidden_checksums = []
        output_checksums = []
        with vector_path.open("w", encoding="ascii") as handle:
            handle.write("C 8 8 64 256 128\n")
            weight_manifest = _write_weights(handle, state)
            for token_idx, trace in enumerate(hw_layer.traces):
                meta = 0x3D00 + token_idx
                handle.write(f"T {token_idx:d} {meta:04x}\n")
                hidden_bits = [float_to_fp16_bits(float(value)) for value in layer_input[token_idx].tolist()]
                for dim, value in enumerate(hidden_bits):
                    handle.write(f"H {dim:x} {value & 0xFFFF:04x}\n")
                for dim, value in enumerate(trace.final_fp32):
                    handle.write(f"O {dim:x} {int(value) & 0xFFFFFFFF:08x}\n")
                hidden_checksums.append(_bits_checksum(hidden_bits, 16))
                output_checksums.append(_bits_checksum([int(value) for value in trace.final_fp32], 32))
        top_info = _token_top_info(hw_model["logits"], tokenizer.eos_id)
        case_manifest = {
            "stage": "ML-M3",
            "length": length,
            "source": "Q2 deterministic trace seed reconstructed from ML-Q2 benchmark rule",
            "prompt_seed": "Once upon a time there was a small red bird who liked kind stories in the garden.",
            "token_ids": [int(idx) for idx in input_ids[0].tolist()],
            "position_ids": position_ids,
            "vector_file": str(vector_path),
            "vector_sha256": sha256_file(vector_path),
            "expected_valid_seq_len": length,
            "weight_manifest": weight_manifest,
            "token_hidden_fp16_sha256": hidden_checksums,
            "expected_layer_output_fp32_sha256": output_checksums,
            "node_checksums_last_token": _node_checksums(hw_layer.traces[-1]),
            "top_info_hardware_aware": top_info,
            "k_cache_checksum": _bits_checksum(_flatten(hw_layer.k_cache), 16),
            "v_cache_checksum": _bits_checksum(_flatten(hw_layer.v_cache), 16),
        }
        write_json(case_dir / "case_manifest.json", case_manifest)
        case_records.append(case_manifest)
        reference_records[f"len_{length}"] = compare_reference_chain(model, input_ids)
        if length in NEXT_TOKEN_LENGTHS:
            next_token_records[f"len_{length}"] = top_info
    manifest = {
        "stage": "ML-M3",
        "required_lengths": VECTOR_LENGTHS_REQUIRED,
        "extended_lengths": VECTOR_LENGTHS_EXTENDED,
        "case_count": len(case_records),
        "cases": case_records,
    }
    write_json(ARTIFACT_ROOT / "manifests" / "vector_manifest.json", manifest)
    write_json(ARTIFACT_ROOT / "comparisons" / "reference_chain.json", {"stage": "ML-M3", "comparisons": reference_records})
    write_json(ARTIFACT_ROOT / "comparisons" / "next_token_reference.json", {"stage": "ML-M3", "cases": next_token_records})
    _write_reports(manifest, reference_records, next_token_records)
    return manifest


def _write_reports(manifest: dict[str, Any], reference_records: dict[str, Any], next_token_records: dict[str, Any]) -> None:
    reports = Path("reports/ml_m3")
    reports.mkdir(parents=True, exist_ok=True)
    ref_lines = [
        "# ML-M3 Reference Chain",
        "",
        "Reference 0 is PyTorch FP32, Reference 1 is FP16-weight PyTorch, and Reference 2 is the existing hardware-aware bit model.",
        "",
        "| Case | FP32 vs FP16 max abs | FP32 vs bit max abs | FP16 vs bit max abs | Bit top-1 agreement |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in reference_records.items():
        ref_lines.append(
            f"| {name} | {row['pytorch_fp32_vs_fp16_weight']['max_abs_error']:.6g} | "
            f"{row['pytorch_fp32_vs_hardware_aware']['max_abs_error']:.6g} | "
            f"{row['fp16_weight_vs_hardware_aware']['max_abs_error']:.6g} | "
            f"{row['pytorch_fp32_vs_hardware_aware']['top1_agreement']:.3f} |"
        )
    ref_lines.append("")
    (reports / "reference_chain.md").write_text("\n".join(ref_lines), encoding="utf-8")
    vector_lines = [
        "# ML-M3 Real-Input Vector Generation",
        "",
        "| Length | Vector | SHA256 | Expected valid_seq_len | Top-1 | EOS rank |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for case in manifest["cases"]:
        vector_lines.append(
            f"| {case['length']} | `{case['vector_file']}` | `{case['vector_sha256']}` | "
            f"{case['expected_valid_seq_len']} | {case['top_info_hardware_aware']['top1']} | "
            f"{case['top_info_hardware_aware']['eos_rank']} |"
        )
    vector_lines.append("")
    (reports / "rtl_smoke_results.md").write_text(
        "# ML-M3 RTL Smoke Results\n\nRTL smoke has not run yet; this file will be updated by the VCS runner.\n",
        encoding="utf-8",
    )
    (reports / "incremental_kv_results.md").write_text(
        "# ML-M3 Incremental KV Results\n\nRTL incremental runs have not run yet; this file will be updated by the VCS runner.\n",
        encoding="utf-8",
    )
    (reports / "next_token_results.md").write_text(
        "# ML-M3 Next-Token Reference\n\n"
        + "\n".join(
            f"- {name}: top1={row['top1']} top5={row['top5']} eos_rank={row['eos_rank']}"
            for name, row in next_token_records.items()
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", action="append", type=int, help="Specific length to generate; may be repeated.")
    args = parser.parse_args()
    payload = generate_vectors(args.length)
    print(json.dumps({"case_count": payload["case_count"], "lengths": [case["length"] for case in payload["cases"]]}, indent=2))


if __name__ == "__main__":
    main()
