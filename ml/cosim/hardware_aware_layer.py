"""Run the accepted Stage 7 bit model from an ML-M2 PyTorch layer."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ml.cosim.fp16_policy import float_to_fp16_bits, fp32_bits_to_float, tensor_to_fp16_bits
from model.projection.projection_reference import WK, WO, WQ, WV
from model.transformer.transformer_layer_reference import TransformerLayerReference


@dataclass
class HardwareAwareLayerResult:
    layer_output: torch.Tensor
    traces: list
    k_cache: list
    v_cache: list


def _weights_from_model(model) -> tuple[dict[int, list], list[int], list[int], list, list]:
    state = model.state_dict()
    weights = {
        WQ: tensor_to_fp16_bits(state["layers.0.attn.wq.weight"]),
        WK: tensor_to_fp16_bits(state["layers.0.attn.wk.weight"]),
        WV: tensor_to_fp16_bits(state["layers.0.attn.wv.weight"]),
        WO: tensor_to_fp16_bits(state["layers.0.attn.wo.weight"]),
    }
    gamma1 = tensor_to_fp16_bits(state["layers.0.norm1.weight"])
    gamma2 = tensor_to_fp16_bits(state["layers.0.norm2.weight"])
    w1 = tensor_to_fp16_bits(state["layers.0.ffn.w1.weight"])
    w2 = tensor_to_fp16_bits(state["layers.0.ffn.w2.weight"])
    return weights, gamma1, gamma2, w1, w2


def build_transformer_layer_reference(model, max_seq_len: int | None = None, pe_num: int = 8) -> TransformerLayerReference:
    cfg = model.config
    weights, gamma1, gamma2, w1, w2 = _weights_from_model(model)
    return TransformerLayerReference(
        n_head=cfg.num_attention_heads,
        d_head=cfg.d_head,
        max_seq_len=max_seq_len or cfg.context_length,
        pe_num=pe_num,
        mha_weights=weights,
        gamma1_fp16=gamma1,
        gamma2_fp16=gamma2,
        w1=w1,
        w2=w2,
    )


def run_hardware_aware_layer(model, layer_inputs: torch.Tensor, pe_num: int = 8) -> HardwareAwareLayerResult:
    """Run Stage 7 bit model token by token.

    `layer_inputs` is `[seq, d_model]` and already includes software-side token
    plus position embeddings.
    """

    ref = build_transformer_layer_reference(model, max_seq_len=model.config.context_length, pe_num=pe_num)
    outputs = []
    traces = []
    for token_index, hidden in enumerate(layer_inputs.detach().cpu().float()):
        hidden_fp16 = [float_to_fp16_bits(float(value)) for value in hidden.tolist()]
        trace = ref.run_token(hidden_fp16, meta=token_index)
        outputs.append([fp32_bits_to_float(value) for value in trace.final_fp32])
        traces.append(trace)
    k_cache, v_cache = ref.mha.stage5.cache.snapshot()
    return HardwareAwareLayerResult(
        layer_output=torch.tensor(outputs, dtype=torch.float32),
        traces=traces,
        k_cache=k_cache,
        v_cache=v_cache,
    )

