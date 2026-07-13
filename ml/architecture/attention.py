"""Standard MHA with append-only KV cache for ML-M2."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ml.architecture.config import HardwareMatchedConfig


@dataclass
class KVCache:
    k: torch.Tensor
    v: torch.Tensor

    @property
    def valid_seq_len(self) -> int:
        return int(self.k.shape[2])

    def reset(self) -> None:
        self.k = self.k[:, :, :0, :]
        self.v = self.v[:, :, :0, :]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: HardwareMatchedConfig):
        super().__init__()
        config.validate()
        self.config = config
        d = config.d_model
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.config.num_attention_heads, self.config.d_head).transpose(1, 2)

    @staticmethod
    def _causal_mask(query_len: int, key_len: int, past_len: int, device: torch.device) -> torch.Tensor:
        query_pos = torch.arange(past_len, past_len + query_len, device=device).view(query_len, 1)
        key_pos = torch.arange(0, key_len, device=device).view(1, key_len)
        return key_pos <= query_pos

    def forward(
        self,
        x: torch.Tensor,
        past_kv: KVCache | None = None,
        use_cache: bool = False,
        need_weights: bool = False,
    ):
        bsz, seq_len, _ = x.shape
        q = self._shape(self.wq(x))
        k_new = self._shape(self.wk(x))
        v_new = self._shape(self.wv(x))
        past_len = 0
        if past_kv is not None:
            past_len = past_kv.valid_seq_len
            k = torch.cat([past_kv.k, k_new], dim=2)
            v = torch.cat([past_kv.v, v_new], dim=2)
        else:
            k = k_new
            v = v_new

        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.config.d_head ** -0.5)
        mask = self._causal_mask(seq_len, k.shape[2], past_len, x.device)
        scores = scores.masked_fill(~mask.view(1, 1, seq_len, k.shape[2]), torch.finfo(scores.dtype).min)
        probs = F.softmax(scores, dim=-1)
        head_out = torch.matmul(probs, v)
        concat = head_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.config.d_model)
        out = self.wo(concat)
        new_cache = KVCache(k.detach(), v.detach()) if use_cache else None
        if need_weights:
            return out, new_cache, {"q": q, "k": k, "v": v, "scores": scores, "probabilities": probs, "head_output": head_out}
        return out, new_cache

