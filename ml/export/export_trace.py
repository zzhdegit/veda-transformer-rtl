"""Trace export for ML-M2 PyTorch and hardware-aware paths."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from ml.cosim.hardware_aware_model import run_hardware_aware_model


def _checksum_tensor(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def tensor_record(name: str, tensor: torch.Tensor, stage: str, token_index: int = -1, layer_index: int = -1) -> dict:
    return {
        "name": name,
        "stage": stage,
        "token_index": token_index,
        "layer_index": layer_index,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "checksum": _checksum_tensor(tensor),
    }


def int_record(name: str, values, stage: str, token_index: int = -1, layer_index: int = -1) -> dict:
    flat: list[int] = []

    def visit(obj) -> None:
        if isinstance(obj, (list, tuple)):
            for item in obj:
                visit(item)
        else:
            flat.append(int(obj))

    visit(values)
    tensor = torch.tensor(flat, dtype=torch.int64)
    return tensor_record(name, tensor, stage, token_index=token_index, layer_index=layer_index)


@torch.no_grad()
def export_trace(model, input_ids: torch.Tensor, output_path: str | Path) -> dict:
    model.eval()
    pytorch = model(input_ids, return_trace=True)
    hw = run_hardware_aware_model(model, input_ids)
    trace = pytorch["trace"]
    records = [
        tensor_record("token_ids", trace["token_ids"], "model"),
        tensor_record("position_ids", trace["position_ids"], "model"),
        tensor_record("token_embedding", trace["token_embedding"], "model"),
        tensor_record("position_embedding", trace["position_embedding"], "model"),
        tensor_record("layer_input", trace["layer_input"], "model", layer_index=0),
        tensor_record("layer_output", trace["layer_output"], "model", layer_index=0),
        tensor_record("final_norm", trace["final_norm"], "model"),
        tensor_record("logits", trace["logits"], "model"),
        tensor_record("hardware_layer_output", hw["layer_output"], "hardware_aware", layer_index=0),
    ]
    layer_trace = trace["layer_0"]
    for key in [
        "rmsnorm1_input",
        "rmsnorm1_output",
        "wo_output",
        "residual1",
        "rmsnorm2",
        "w1_output",
        "relu_output",
        "w2_output",
        "residual2",
    ]:
        records.append(tensor_record(key, layer_trace[key], "layer", layer_index=0))
    attention = layer_trace["attention"]
    for key, name in [
        ("q", "q_projection_fp32"),
        ("k", "k_projection_fp32"),
        ("v", "v_projection_fp32"),
        ("scores", "scaled_score"),
        ("probabilities", "softmax_probability"),
        ("head_output", "per_head_output"),
    ]:
        records.append(tensor_record(name, attention[key], "layer", layer_index=0))
    q_fp16 = []
    k_fp16 = []
    v_fp16 = []
    concat_fp32 = []
    concat_fp16 = []
    activation_fp16 = []
    raw_scores = []
    scaled_scores = []
    probabilities = []
    for token_index, bit_trace in enumerate(hw["traces"]):
        q_fp16.append(bit_trace.mha.qkv.q_fp16)
        k_fp16.append(bit_trace.mha.qkv.k_fp16)
        v_fp16.append(bit_trace.mha.qkv.v_fp16)
        concat_fp32.append(bit_trace.mha.concat_fp32)
        concat_fp16.append(bit_trace.mha.concat_fp16)
        activation_fp16.append(bit_trace.activation_fp16)
        raw_scores.append([head.raw_scores for head in bit_trace.mha.attention.attention_traces])
        scaled_scores.append([head.scaled_scores for head in bit_trace.mha.attention.attention_traces])
        probabilities.append([head.probabilities for head in bit_trace.mha.attention.attention_traces])
    records.extend(
        [
            int_record("q_fp16", q_fp16, "hardware_aware", layer_index=0),
            int_record("k_fp16", k_fp16, "hardware_aware", layer_index=0),
            int_record("v_fp16", v_fp16, "hardware_aware", layer_index=0),
            int_record("score", raw_scores, "hardware_aware", layer_index=0),
            int_record("scaled_score_bits", scaled_scores, "hardware_aware", layer_index=0),
            int_record("softmax_probability_bits", probabilities, "hardware_aware", layer_index=0),
            int_record("concat_fp32", concat_fp32, "hardware_aware", layer_index=0),
            int_record("concat_fp16", concat_fp16, "hardware_aware", layer_index=0),
            int_record("activation_fp16", activation_fp16, "hardware_aware", layer_index=0),
            int_record("k_cache_after_token", hw["k_cache"], "cache", layer_index=0),
            int_record("v_cache_after_token", hw["v_cache"], "cache", layer_index=0),
        ]
    )
    cache_records = {
        "valid_seq_len": input_ids.shape[1],
        "k_cache_heads": len(hw["k_cache"]),
        "v_cache_heads": len(hw["v_cache"]),
        "head_index_order": "head-major",
        "dimension_index_order": "dimension-major within each token",
    }
    manifest = {
        "stage": "ML-M2F",
        "records": records,
        "cache": cache_records,
        "trace_node_count": len(records),
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
