"""Optimizer factory."""

from __future__ import annotations

import inspect

import torch


def build_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1.0e-8,
    fused: bool | None = None,
) -> torch.optim.Optimizer:
    kwargs = {
        "lr": learning_rate,
        "weight_decay": weight_decay,
        "betas": betas,
        "eps": eps,
    }
    if fused is not None and "fused" in inspect.signature(torch.optim.AdamW).parameters:
        kwargs["fused"] = fused
    return torch.optim.AdamW(model.parameters(), **kwargs)
