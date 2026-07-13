"""Deterministic prompt suite for ML-M2."""

from __future__ import annotations

from ml.data.fixtures import SMOKE_TEST_PROMPTS


FORMAL_TEST_PROMPTS = [
    "Once upon a time",
    "A little girl found",
    "The cat and the dog",
    "In the small garden",
]


def smoke_prompts() -> list[str]:
    return list(SMOKE_TEST_PROMPTS)


def formal_prompts() -> list[str]:
    return list(FORMAL_TEST_PROMPTS)

