"""Greedy text generation helpers."""

from __future__ import annotations

import torch


@torch.no_grad()
def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 16) -> str:
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], dtype=torch.long)
    generated = model.generate_greedy(input_ids, max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_id)
    return tokenizer.decode(generated[0].tolist())

