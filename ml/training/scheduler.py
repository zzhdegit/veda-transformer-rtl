"""Scheduler factory for ML-M2."""

from __future__ import annotations

import math

import torch


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
    schedule: str = "constant",
):
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-8, float(step + 1) / float(warmup_steps))
        if schedule == "cosine":
            denom = max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, float(step - warmup_steps) / denom))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
