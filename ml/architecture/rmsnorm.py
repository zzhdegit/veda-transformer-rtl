"""RMSNorm layer matching the ML-M2 contract."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1.0e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(variance + self.eps)
        return x * inv_rms * self.weight

