"""Optimizer factory."""

from __future__ import annotations

import torch


def build_optimizer(model: torch.nn.Module, learning_rate: float, weight_decay: float = 0.0) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

