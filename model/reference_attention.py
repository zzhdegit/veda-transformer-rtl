"""Floating-point reference model for single-head generation attention.

This model is intentionally not bit-accurate. It defines the Stage 0
algorithmic behavior and leaves explicit hooks for a later bit-accurate model.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Iterable, List, Sequence, Tuple


Vector = List[float]
Matrix = List[Vector]


@dataclass(frozen=True)
class AttentionTrace:
    raw_scores: Vector
    scaled_scores: Vector
    max_score: float
    exp_sum: float
    probabilities: Vector
    output: Vector


@dataclass(frozen=True)
class Comparison:
    max_raw_score_diff: float
    max_scaled_score_diff: float
    max_probability_diff: float
    max_output_diff: float
    max_score_diff: float
    exp_sum_diff: float
    probability_sum: float

    @property
    def max_diff(self) -> float:
        return max(
            self.max_raw_score_diff,
            self.max_scaled_score_diff,
            self.max_probability_diff,
            self.max_output_diff,
            self.max_score_diff,
            self.exp_sum_diff,
            abs(self.probability_sum - 1.0),
        )


def dot(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    if len(lhs) != len(rhs):
        raise ValueError(f"dot length mismatch: {len(lhs)} != {len(rhs)}")
    return sum(a * b for a, b in zip(lhs, rhs))


def _validate_inputs(q: Sequence[float], k_cache: Sequence[Sequence[float]], v_cache: Sequence[Sequence[float]]) -> None:
    if not q:
        raise ValueError("d_head must be positive")
    if not k_cache:
        raise ValueError("seq_len must be positive")
    if len(k_cache) != len(v_cache):
        raise ValueError("K and V sequence lengths differ")

    d_head = len(q)
    for row_name, matrix in (("K", k_cache), ("V", v_cache)):
        for idx, row in enumerate(matrix):
            if len(row) != d_head:
                raise ValueError(f"{row_name}[{idx}] has dimension {len(row)}, expected {d_head}")


def stable_softmax(values: Sequence[float]) -> Tuple[float, float, Vector]:
    if not values:
        raise ValueError("softmax requires at least one value")
    max_score = max(values)
    exp_values = [math.exp(value - max_score) for value in values]
    exp_sum = sum(exp_values)
    probabilities = [value / exp_sum for value in exp_values]
    return max_score, exp_sum, probabilities


def online_max_exp_sum(values: Iterable[float]) -> Tuple[float, float]:
    """Online softmax reduction for max and exp_sum.

    For each incoming x:

        m_new = max(m_old, x)
        z_new = z_old * exp(m_old - m_new) + exp(x - m_new)
    """

    iterator = iter(values)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("online reduction requires at least one value") from exc

    max_score = first
    exp_sum = 1.0
    for value in iterator:
        new_max = max(max_score, value)
        exp_sum = exp_sum * math.exp(max_score - new_max) + math.exp(value - new_max)
        max_score = new_max
    return max_score, exp_sum


def direct_attention(q: Sequence[float], k_cache: Sequence[Sequence[float]], v_cache: Sequence[Sequence[float]]) -> AttentionTrace:
    """Direct two-pass softmax reference."""

    _validate_inputs(q, k_cache, v_cache)
    d_head = len(q)
    scale = 1.0 / math.sqrt(d_head)
    raw_scores = [dot(q, k_row) for k_row in k_cache]
    scaled_scores = [score * scale for score in raw_scores]
    max_score, exp_sum, probabilities = stable_softmax(scaled_scores)
    output = weighted_sum(probabilities, v_cache, d_head)
    return AttentionTrace(raw_scores, scaled_scores, max_score, exp_sum, probabilities, output)


def online_attention(q: Sequence[float], k_cache: Sequence[Sequence[float]], v_cache: Sequence[Sequence[float]]) -> AttentionTrace:
    """Reference using online max/exp_sum reduction plus a normalization pass."""

    _validate_inputs(q, k_cache, v_cache)
    d_head = len(q)
    scale = 1.0 / math.sqrt(d_head)
    raw_scores = [dot(q, k_row) for k_row in k_cache]
    scaled_scores = [score * scale for score in raw_scores]
    max_score, exp_sum = online_max_exp_sum(scaled_scores)
    probabilities = [math.exp(score - max_score) / exp_sum for score in scaled_scores]
    output = weighted_sum(probabilities, v_cache, d_head)
    return AttentionTrace(raw_scores, scaled_scores, max_score, exp_sum, probabilities, output)


def weighted_sum(probabilities: Sequence[float], values: Sequence[Sequence[float]], d_head: int) -> Vector:
    output = [0.0 for _ in range(d_head)]
    for probability, row in zip(probabilities, values):
        for dim in range(d_head):
            output[dim] += probability * row[dim]
    return output


def compare_traces(lhs: AttentionTrace, rhs: AttentionTrace) -> Comparison:
    return Comparison(
        max_raw_score_diff=max_abs_diff(lhs.raw_scores, rhs.raw_scores),
        max_scaled_score_diff=max_abs_diff(lhs.scaled_scores, rhs.scaled_scores),
        max_probability_diff=max_abs_diff(lhs.probabilities, rhs.probabilities),
        max_output_diff=max_abs_diff(lhs.output, rhs.output),
        max_score_diff=abs(lhs.max_score - rhs.max_score),
        exp_sum_diff=abs(lhs.exp_sum - rhs.exp_sum),
        probability_sum=sum(rhs.probabilities),
    )


def max_abs_diff(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    if len(lhs) != len(rhs):
        raise ValueError(f"length mismatch: {len(lhs)} != {len(rhs)}")
    if not lhs:
        return 0.0
    return max(abs(a - b) for a, b in zip(lhs, rhs))


def generate_random_inputs(d_head: int, seq_len: int, seed: int) -> Tuple[Vector, Matrix, Matrix]:
    if d_head <= 0:
        raise ValueError("d_head must be positive")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")

    rng = random.Random(seed)
    q = [rng.uniform(-1.0, 1.0) for _ in range(d_head)]
    k_cache = [[rng.uniform(-1.0, 1.0) for _ in range(d_head)] for _ in range(seq_len)]
    v_cache = [[rng.uniform(-1.0, 1.0) for _ in range(d_head)] for _ in range(seq_len)]
    return q, k_cache, v_cache


def run_case(d_head: int, seq_len: int, seed: int) -> Tuple[AttentionTrace, AttentionTrace, Comparison]:
    q, k_cache, v_cache = generate_random_inputs(d_head, seq_len, seed)
    direct = direct_attention(q, k_cache, v_cache)
    online = online_attention(q, k_cache, v_cache)
    return direct, online, compare_traces(direct, online)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Stage 0 single-head attention reference model.")
    parser.add_argument("--d-head", type=int, default=8, help="Attention head dimension.")
    parser.add_argument("--seq-len", type=int, default=32, help="Active sequence length.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--check", action="store_true", help="Fail if direct and online references differ.")
    parser.add_argument("--tolerance", type=float, default=1e-12, help="Comparison tolerance for --check.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    direct, online, comparison = run_case(args.d_head, args.seq_len, args.seed)
    if args.check and comparison.max_diff > args.tolerance:
        raise SystemExit(f"direct and online attention differ: {comparison}")

    print(
        json.dumps(
            {
                "d_head": args.d_head,
                "seq_len": args.seq_len,
                "seed": args.seed,
                "direct": asdict(direct),
                "online": asdict(online),
                "comparison": asdict(comparison),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
