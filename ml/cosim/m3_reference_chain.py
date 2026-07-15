"""Reference-chain utilities for ML-M3."""

from __future__ import annotations

import torch

from ml.cosim.fp16_policy import fp16_bits_nested_to_tensor
from ml.cosim.hardware_aware_model import run_hardware_aware_model
from ml.evaluation.evaluate_quantization import logits_agreement, tensor_error_metrics
from ml.export.formal_export import _fp16_weight_rounded_model


@torch.no_grad()
def compare_reference_chain(model, input_ids: torch.Tensor) -> dict:
    """Compare PyTorch FP32, exported-FP16-weight PyTorch, and bit model."""

    model.eval()
    fp16_model = _fp16_weight_rounded_model(model)
    fp32 = model(input_ids, return_trace=True)
    fp16 = fp16_model(input_ids, return_trace=True)
    bit = run_hardware_aware_model(model, input_ids)
    fp32_logits = fp32["logits"]
    fp16_logits = fp16["logits"]
    bit_logits = bit["logits"]
    pt_k = fp32["trace"]["layer_0"]["attention"]["k"][0].detach().float()
    pt_v = fp32["trace"]["layer_0"]["attention"]["v"][0].detach().float()
    bit_k = fp16_bits_nested_to_tensor(bit["k_cache"])
    bit_v = fp16_bits_nested_to_tensor(bit["v_cache"])
    return {
        "pytorch_fp32_vs_fp16_weight": {
            **tensor_error_metrics(fp32_logits, fp16_logits),
            **logits_agreement(fp32_logits, fp16_logits),
        },
        "pytorch_fp32_vs_hardware_aware": {
            **tensor_error_metrics(fp32_logits, bit_logits),
            **logits_agreement(fp32_logits, bit_logits),
        },
        "fp16_weight_vs_hardware_aware": {
            **tensor_error_metrics(fp16_logits, bit_logits),
            **logits_agreement(fp16_logits, bit_logits),
        },
        "layer_output": {
            "pytorch_fp32_vs_hardware_aware": tensor_error_metrics(fp32["trace"]["layer_output"], bit["layer_output"]),
            "fp16_weight_vs_hardware_aware": tensor_error_metrics(fp16["trace"]["layer_output"], bit["layer_output"]),
        },
        "kv_cache": {
            "k": tensor_error_metrics(pt_k, bit_k),
            "v": tensor_error_metrics(pt_v, bit_v),
        },
    }
