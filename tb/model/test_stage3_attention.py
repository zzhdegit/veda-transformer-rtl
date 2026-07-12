import math

from model.attention.single_head_cycle_model import estimate_attention_cycles
from model.attention.single_head_reference import (
    single_head_attention_bit_model,
    single_head_high_precision,
)
from model.attention.softmax_reference import (
    fp32_exp,
    fp32_to_float,
    normalize_scores,
    online_softmax_reduction,
    probability_sum_float,
)


FP16_ONE = 0x3C00
FP16_TWO = 0x4000
FP16_HALF = 0x3800
FP16_NEG_ONE = 0xBC00
FP16_NEG_HALF = 0xB800
FP16_ZERO = 0x0000


def test_exp_directed_values_and_clamp():
    cases = {
        0x00000000: 0x3F800000,
        0xBA83126F: 0x3F7FBE7F,
        0xBDCCCCCD: 0x3F67A36D,
        0xBF800000: 0x3EBC5AB2,
        0xC0A00000: 0x3BDCC9FF,
        0xC1200000: 0x383E6BCE,
        0xC1A00000: 0x310DA433,
        0xC1A80000: 0x00000000,
    }
    for inp, expected in cases.items():
        assert fp32_exp(inp) == expected


def test_online_reduction_uniform_scores():
    scores = [0x3F000000 for _ in range(8)]
    reduction = online_softmax_reduction(scores)
    normalization = normalize_scores(scores, reduction["max"], reduction["exp_sum"])
    assert reduction["max"] == 0x3F000000
    assert abs(probability_sum_float(normalization["probabilities"]) - 1.0) < 2e-7
    for probability in normalization["probabilities"]:
        assert abs(fp32_to_float(probability) - 0.125) < 2e-7


def test_attention_bit_model_seq_lengths_and_dimensions():
    dims = [1, 7, 8, 9, 13, 16]
    seqs = [1, 2, 3, 7, 8, 15, 31, 32]
    pattern_q = [FP16_ONE, FP16_HALF, FP16_NEG_ONE, FP16_TWO, FP16_NEG_HALF]
    pattern_k = [FP16_TWO, FP16_NEG_ONE, FP16_HALF, FP16_ONE, FP16_ZERO]
    pattern_v = [FP16_HALF, FP16_ONE, FP16_NEG_HALF, FP16_TWO, FP16_NEG_ONE]
    for d_head in dims:
        q = [pattern_q[idx % len(pattern_q)] for idx in range(d_head)]
        for seq_len in seqs:
            k = [[pattern_k[(tok + dim) % len(pattern_k)] for dim in range(d_head)] for tok in range(seq_len)]
            v = [[pattern_v[(2 * tok + dim) % len(pattern_v)] for dim in range(d_head)] for tok in range(seq_len)]
            trace = single_head_attention_bit_model(q, k, v, pe_num=8)
            high = single_head_high_precision(q, k, v)
            assert len(trace.raw_scores) == seq_len
            assert len(trace.probabilities) == seq_len
            assert len(trace.output) == d_head
            assert abs(sum(high["probabilities"]) - 1.0) < 1e-12
            assert math.isfinite(high["exp_sum"])


def test_one_hot_like_and_zero_v_cases():
    q = [FP16_TWO] * 8
    k = [[FP16_NEG_ONE] * 8, [FP16_ZERO] * 8, [FP16_TWO] * 8]
    v = [[FP16_ZERO] * 8, [FP16_ONE] * 8, [FP16_TWO] * 8]
    trace = single_head_attention_bit_model(q, k, v, pe_num=8)
    assert len(trace.output) == 8
    assert max(fp32_to_float(p) for p in trace.probabilities) > 0.75

    zero_v = [[FP16_ZERO] * 8 for _ in range(4)]
    zero_trace = single_head_attention_bit_model(q, k + [[FP16_ZERO] * 8], zero_v, pe_num=8)
    assert all(value == 0 for value in zero_trace.output)


def test_cycle_model_records_required_counters():
    estimate = estimate_attention_cycles(seq_len=32, d_head=16, pe_num=8)
    data = estimate.as_dict()
    for key in (
        "total_attention_cycles",
        "qk_cycles",
        "qk_pe_busy_cycles",
        "scale_cycles",
        "reduction_cycles",
        "reduction_finalize_cycles",
        "normalization_cycles",
        "sv_cycles",
        "pe_stall_cycles",
        "sfu_stall_cycles",
        "buffer_stall_cycles",
        "output_stall_cycles",
        "score_buffer_peak_occupancy",
    ):
        assert key in data
    assert estimate.score_buffer_peak_occupancy == 32
