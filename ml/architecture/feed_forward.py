"""Bias-free ReLU FFN for ML-M2."""

from __future__ import annotations

from torch import nn
from torch.nn import functional as F

from ml.architecture.config import HardwareMatchedConfig


class FeedForward(nn.Module):
    def __init__(self, config: HardwareMatchedConfig):
        super().__init__()
        config.validate()
        self.w1 = nn.Linear(config.d_model, config.d_ffn, bias=False)
        self.w2 = nn.Linear(config.d_ffn, config.d_model, bias=False)

    def forward(self, x):
        return self.w2(F.relu(self.w1(x)))

