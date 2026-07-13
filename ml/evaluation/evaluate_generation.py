"""Deterministic generation evaluation."""

from __future__ import annotations

from ml.inference.generate import generate_text


def evaluate_generation(model, tokenizer, prompts: list[str], max_new_tokens: int = 24) -> list[dict[str, str]]:
    return [
        {"prompt": prompt, "generated": generate_text(model, tokenizer, prompt, max_new_tokens=max_new_tokens)}
        for prompt in prompts
    ]

