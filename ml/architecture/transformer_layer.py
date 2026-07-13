"""One hardware-matched Pre-Norm Transformer layer."""

from __future__ import annotations

from torch import nn

from ml.architecture.attention import KVCache, MultiHeadSelfAttention
from ml.architecture.config import HardwareMatchedConfig
from ml.architecture.feed_forward import FeedForward
from ml.architecture.rmsnorm import RMSNorm


class TransformerLayer(nn.Module):
    def __init__(self, config: HardwareMatchedConfig):
        super().__init__()
        config.validate()
        self.norm1 = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attn = MultiHeadSelfAttention(config)
        self.norm2 = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn = FeedForward(config)

    def forward(self, x, past_kv: KVCache | None = None, use_cache: bool = False, return_trace: bool = False):
        n1 = self.norm1(x)
        if return_trace:
            a, new_cache, attn_trace = self.attn(n1, past_kv=past_kv, use_cache=use_cache, need_weights=True)
        else:
            a, new_cache = self.attn(n1, past_kv=past_kv, use_cache=use_cache)
            attn_trace = None
        r1 = x + a
        n2 = self.norm2(r1)
        h1 = self.ffn.w1(n2)
        h = h1.relu()
        f = self.ffn.w2(h)
        y = r1 + f
        if return_trace:
            return y, new_cache, {
                "rmsnorm1_input": x,
                "rmsnorm1_output": n1,
                "attention": attn_trace,
                "wo_output": a,
                "residual1": r1,
                "rmsnorm2": n2,
                "w1_output": h1,
                "relu_output": h,
                "w2_output": f,
                "residual2": y,
            }
        return y, new_cache

