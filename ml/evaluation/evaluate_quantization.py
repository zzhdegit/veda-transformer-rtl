"""Quantization comparison metrics."""

from __future__ import annotations

import torch


def tensor_error_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = a.detach().float()
    b = b.detach().float()
    diff = (a - b).abs()
    denom = torch.linalg.norm(a) * torch.linalg.norm(b)
    cosine = float(torch.sum(a * b) / denom) if float(denom) != 0.0 else 1.0
    return {
        "max_abs_error": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_error": float(diff.mean().item()) if diff.numel() else 0.0,
        "cosine_similarity": cosine,
    }


def logits_agreement(fp32_logits: torch.Tensor, test_logits: torch.Tensor, k: int = 5) -> dict:
    top1_a = torch.argmax(fp32_logits, dim=-1)
    top1_b = torch.argmax(test_logits, dim=-1)
    topk_a = torch.topk(fp32_logits, k=min(k, fp32_logits.shape[-1]), dim=-1).indices
    topk_b = torch.topk(test_logits, k=min(k, test_logits.shape[-1]), dim=-1).indices
    overlap = []
    for lhs, rhs in zip(topk_a.reshape(-1, topk_a.shape[-1]), topk_b.reshape(-1, topk_b.shape[-1])):
        overlap.append(len(set(lhs.tolist()) & set(rhs.tolist())) / float(lhs.numel()))
    diff_positions = torch.nonzero(top1_a.reshape(-1) != top1_b.reshape(-1)).reshape(-1)
    return {
        "top1_agreement": float((top1_a == top1_b).float().mean().item()),
        "top5_overlap": float(sum(overlap) / len(overlap)) if overlap else 1.0,
        "first_differing_token": int(diff_positions[0].item()) if diff_positions.numel() else -1,
    }

