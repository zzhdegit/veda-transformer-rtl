"""Hardware-matched one-layer causal LM."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ml.architecture.attention import KVCache
from ml.architecture.config import HardwareMatchedConfig
from ml.architecture.rmsnorm import RMSNorm
from ml.architecture.transformer_layer import TransformerLayer


class HardwareMatchedCausalLM(nn.Module):
    def __init__(self, config: HardwareMatchedConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.layers = nn.ModuleList([TransformerLayer(config)])
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def _position_ids(self, batch_size: int, seq_len: int, start_pos: int, device) -> torch.Tensor:
        if start_pos + seq_len > self.config.context_length:
            raise ValueError("position exceeds context_length")
        positions = torch.arange(start_pos, start_pos + seq_len, device=device)
        return positions.view(1, seq_len).expand(batch_size, seq_len)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        past_kv: list[KVCache | None] | None = None,
        use_cache: bool = False,
        start_pos: int = 0,
        return_trace: bool = False,
    ):
        bsz, seq_len = input_ids.shape
        pos = self._position_ids(bsz, seq_len, start_pos, input_ids.device)
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(pos)
        x = token_emb + pos_emb
        new_caches: list[KVCache | None] = []
        traces = {
            "token_ids": input_ids,
            "position_ids": pos,
            "token_embedding": token_emb,
            "position_embedding": pos_emb,
            "layer_input": x,
        }
        layer_past = past_kv or [None for _ in self.layers]
        for idx, layer in enumerate(self.layers):
            if return_trace:
                x, cache, layer_trace = layer(x, past_kv=layer_past[idx], use_cache=use_cache, return_trace=True)
                traces[f"layer_{idx}"] = layer_trace
            else:
                x, cache = layer(x, past_kv=layer_past[idx], use_cache=use_cache)
            new_caches.append(cache)
        layer_output = x
        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
        if return_trace:
            traces["layer_output"] = layer_output
            traces["final_norm"] = x
            traces["logits"] = logits
            top_values, top_indices = torch.topk(logits[:, -1, :], k=min(5, logits.shape[-1]), dim=-1)
            traces["top_k"] = {"values": top_values, "indices": top_indices}
            traces["next_token"] = torch.argmax(logits[:, -1, :], dim=-1)
        return {"logits": logits, "loss": loss, "past_kv": new_caches if use_cache else None, "trace": traces if return_trace else None}

    @torch.no_grad()
    def generate_greedy(self, input_ids: torch.Tensor, max_new_tokens: int = 16, eos_token_id: int | None = None) -> torch.Tensor:
        self.eval()
        generated = input_ids.clone()
        cache = None
        for step in range(max_new_tokens):
            if step == 0:
                current = generated
                start_pos = 0
            else:
                current = generated[:, -1:]
                start_pos = generated.shape[1] - 1
            out = self(current, past_kv=cache, use_cache=True, start_pos=start_pos)
            cache = out["past_kv"]
            next_token = torch.argmax(out["logits"][:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break
        return generated

