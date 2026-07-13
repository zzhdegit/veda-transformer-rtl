"""Incremental decode comparison helpers."""

from __future__ import annotations

import torch


@torch.no_grad()
def compare_full_vs_incremental(model, input_ids: torch.Tensor, atol: float = 1e-5) -> dict:
    model.eval()
    full = model(input_ids)["logits"]
    cache = None
    parts = []
    for pos in range(input_ids.shape[1]):
        out = model(input_ids[:, pos:pos + 1], past_kv=cache, use_cache=True, start_pos=pos)
        cache = out["past_kv"]
        parts.append(out["logits"])
    incremental = torch.cat(parts, dim=1)
    diff = (full - incremental).abs()
    return {
        "max_abs_error": float(diff.max().item()),
        "allclose": bool(torch.allclose(full, incremental, atol=atol)),
        "valid_seq_len": int(cache[0].valid_seq_len) if cache else 0,
    }

