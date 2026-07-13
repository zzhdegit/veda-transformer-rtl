"""Model-level hardware-aware path for ML-M2."""

from __future__ import annotations

import torch

from ml.cosim.hardware_aware_layer import run_hardware_aware_layer


@torch.no_grad()
def run_hardware_aware_model(model, input_ids: torch.Tensor) -> dict:
    model.eval()
    bsz, seq_len = input_ids.shape
    if bsz != 1:
        raise ValueError("hardware-aware ML-M2 path currently supports batch size 1")
    pos = model._position_ids(bsz, seq_len, 0, input_ids.device)
    token_emb = model.token_embedding(input_ids)
    pos_emb = model.position_embedding(pos)
    layer_input = (token_emb + pos_emb)[0].detach().cpu()
    layer_result = run_hardware_aware_layer(model, layer_input)
    layer_output = layer_result.layer_output.unsqueeze(0)
    final_norm = model.final_norm(layer_output)
    logits = model.lm_head(final_norm)
    return {
        "logits": logits,
        "layer_input": layer_input,
        "layer_output": layer_output,
        "k_cache": layer_result.k_cache,
        "v_cache": layer_result.v_cache,
        "k_cache_history": layer_result.k_cache_history,
        "v_cache_history": layer_result.v_cache_history,
        "traces": layer_result.traces,
    }
