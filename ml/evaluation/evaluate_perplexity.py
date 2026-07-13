"""Perplexity helper."""

from __future__ import annotations

import math


def perplexity_from_loss(loss: float) -> float:
    return math.exp(loss) if math.isfinite(loss) and loss < 50 else float("inf")

