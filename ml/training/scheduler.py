"""Scheduler factory for ML-M2."""

from __future__ import annotations

import torch


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int = 0):
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-8, float(step + 1) / float(warmup_steps))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

