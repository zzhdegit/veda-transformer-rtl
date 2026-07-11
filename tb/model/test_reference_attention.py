from __future__ import annotations

import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.reference_attention import compare_traces, direct_attention, online_attention, run_case


TOLERANCE = 1e-12


def assert_traces_close(direct, online, tolerance=TOLERANCE):
    comparison = compare_traces(direct, online)
    assert comparison.max_raw_score_diff <= tolerance
    assert comparison.max_scaled_score_diff <= tolerance
    assert comparison.max_probability_diff <= tolerance
    assert comparison.max_output_diff <= tolerance
    assert comparison.max_score_diff <= tolerance
    assert comparison.exp_sum_diff <= tolerance
    assert abs(comparison.probability_sum - 1.0) <= tolerance


def test_seq_len_one():
    q = [0.25, -0.5, 0.75, 1.0]
    k_cache = [[1.0, 2.0, -1.0, 0.5]]
    v_cache = [[0.1, 0.2, 0.3, 0.4]]

    direct = direct_attention(q, k_cache, v_cache)
    online = online_attention(q, k_cache, v_cache)

    assert_traces_close(direct, online)
    assert direct.probabilities == [1.0]
    assert direct.output == v_cache[0]


def test_all_scores_equal():
    q = [1.0, -2.0, 0.5, 3.0]
    k_row = [0.25, 0.25, 0.25, 0.25]
    k_cache = [list(k_row) for _ in range(5)]
    v_cache = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ]

    direct = direct_attention(q, k_cache, v_cache)
    online = online_attention(q, k_cache, v_cache)

    assert_traces_close(direct, online)
    for probability in direct.probabilities:
        assert abs(probability - 0.2) <= TOLERANCE


def test_one_score_much_larger():
    q = [1.0, 0.0, 0.0, 0.0]
    k_cache = [
        [0.0, 0.0, 0.0, 0.0],
        [40.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
    ]
    v_cache = [
        [0.0, 0.0, 0.0, 0.0],
        [3.0, -2.0, 1.0, 4.0],
        [1.0, 1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0, -1.0],
    ]

    direct = direct_attention(q, k_cache, v_cache)
    online = online_attention(q, k_cache, v_cache)

    assert_traces_close(direct, online)
    assert direct.probabilities[1] > 0.999999
    for got, expected in zip(direct.output, v_cache[1]):
        assert abs(got - expected) < 1e-5


def test_random_inputs_online_matches_direct():
    for seed in range(10):
        direct, online, comparison = run_case(d_head=8, seq_len=32, seed=seed)
        assert_traces_close(direct, online)
        assert comparison.max_diff <= TOLERANCE


def test_probability_sum_close_to_one():
    direct, online, _ = run_case(d_head=8, seq_len=17, seed=123)

    assert math.isclose(sum(direct.probabilities), 1.0, rel_tol=0.0, abs_tol=TOLERANCE)
    assert math.isclose(sum(online.probabilities), 1.0, rel_tol=0.0, abs_tol=TOLERANCE)


def test_large_dynamic_range_is_stable():
    q = [1000.0]
    k_cache = [[-2.0], [0.0], [1.5], [2.0]]
    v_cache = [[-1.0], [0.25], [4.0], [7.0]]

    direct = direct_attention(q, k_cache, v_cache)
    online = online_attention(q, k_cache, v_cache)

    assert_traces_close(direct, online)
    for trace in (direct, online):
        assert math.isfinite(trace.max_score)
        assert math.isfinite(trace.exp_sum)
        assert all(math.isfinite(value) for value in trace.probabilities)
        assert all(math.isfinite(value) for value in trace.output)
        assert math.isclose(sum(trace.probabilities), 1.0, rel_tol=0.0, abs_tol=TOLERANCE)
    assert direct.probabilities[-1] > 1.0 - TOLERANCE
    assert abs(direct.output[0] - v_cache[-1][0]) <= TOLERANCE


def test_current_kv_append_order_uses_updated_cache():
    q = [1.0, 0.0]
    past_k = [[0.0, 1.0], [1.0, 0.0]]
    past_v = [[-1.0, -1.0], [1.0, 1.0]]
    current_k = [3.0, 0.0]
    current_v = [9.0, 9.0]

    past_only = direct_attention(q, past_k, past_v)
    updated_cache = direct_attention(q, past_k + [current_k], past_v + [current_v])

    assert len(updated_cache.probabilities) == len(past_k) + 1
    assert updated_cache.probabilities[-1] > max(updated_cache.probabilities[:-1])
    assert updated_cache.output != past_only.output


def _run_fallback() -> int:
    tests = [(name, obj) for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
    failures = []
    for name, test in tests:
        try:
            test()
        except Exception as exc:  # pragma: no cover - fallback runner path
            failures.append((name, exc))

    if failures:
        for name, exc in failures:
            print(f"FAIL {name}: {exc}")
        return 1

    print(f"PASS {len(tests)} tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_fallback())
