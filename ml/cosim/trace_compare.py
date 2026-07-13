"""Comparison helpers for PyTorch and hardware-aware paths."""

from __future__ import annotations

import torch

from ml.evaluation.evaluate_quantization import logits_agreement, tensor_error_metrics


def compare_logits(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    metrics = tensor_error_metrics(reference, candidate)
    metrics.update(logits_agreement(reference, candidate))
    return metrics

