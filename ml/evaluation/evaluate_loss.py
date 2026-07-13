"""Evaluate average causal LM loss."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_loss(model, dataset, batch_size: int = 8, device: str = "cpu") -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval().to(device)
    losses = []
    for input_ids, labels in loader:
        out = model(input_ids.to(device), labels=labels.to(device))
        losses.append(float(out["loss"].detach().cpu()))
    return sum(losses) / len(losses) if losses else float("nan")

