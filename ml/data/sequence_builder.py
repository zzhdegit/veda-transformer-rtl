"""Causal LM sequence packing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class SequenceBatch:
    input_ids: list[list[int]]
    labels: list[list[int]]


def deterministic_split(items: Sequence[str], validation_fraction: float = 0.1, test_count: int = 3):
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    items = list(items)
    test = items[:test_count]
    remaining = items[test_count:]
    val_count = max(1, int(len(remaining) * validation_fraction)) if remaining else 0
    validation = remaining[:val_count]
    train = remaining[val_count:]
    return train, validation, test


def build_lm_sequences(
    token_ids: Sequence[int],
    context_length: int,
    pad_id: int,
    stride: int | None = None,
    min_real_tokens: int = 2,
) -> SequenceBatch:
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if stride is None:
        stride = context_length
    if stride <= 0:
        raise ValueError("stride must be positive")
    ids = list(token_ids)
    inputs: list[list[int]] = []
    labels: list[list[int]] = []
    need = context_length + 1
    for start in range(0, max(len(ids) - 1, 1), stride):
        window = ids[start:start + need]
        if len(window) < min_real_tokens:
            continue
        if len(window) < need:
            window = window + [pad_id] * (need - len(window))
        label_window = [token if token != pad_id else -100 for token in window[1:]]
        inputs.append(window[:-1])
        labels.append(label_window)
        if start + need >= len(ids):
            break
    return SequenceBatch(inputs, labels)
