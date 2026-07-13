from model.attention.paper_attention_reference import (
    compare_paper_attention_to_legacy,
    paper_single_head_attention_bit_model,
)
from model.attention.paper_attention_cycle_model import estimate_paper_attention_cycles


FP16_ZERO = 0x0000
FP16_ONE = 0x3C00
FP16_NEG_ONE = 0xBC00
FP16_TWO = 0x4000
FP16_HALF = 0x3800
FP16_NEG_HALF = 0xB800


def deterministic_values(length):
    pool = [FP16_ONE, FP16_NEG_ONE, FP16_TWO, FP16_HALF, FP16_NEG_HALF, 0x3400, 0xB400, 0x3000]
    return [pool[(index * 3 + 1) % len(pool)] for index in range(length)]


def make_attention_case(d_head, seq_len):
    q = deterministic_values(d_head)
    k = []
    v = []
    for token in range(seq_len):
        row = deterministic_values(d_head)
        if token % 2:
            row = list(reversed(row))
        k.append(row)
        v.append(row[token % d_head :] + row[: token % d_head])
    return q, k, v


def test_paper_attention_matches_legacy_for_stage8_dimensions():
    for d_head in (8, 16):
        for seq_len in (1, 2, 3, 7, 8):
            q, k, v = make_attention_case(d_head, seq_len)
            comparison = compare_paper_attention_to_legacy(q, k, v)
            assert comparison["raw_scores_bit_exact"]
            assert comparison["scaled_scores_bit_exact"]
            assert comparison["probabilities_bit_exact"]
            assert comparison["output_bit_exact"]


def test_paper_attention_dense_mixed_and_cancellation():
    q = [FP16_ONE, FP16_ONE, FP16_NEG_ONE, FP16_NEG_ONE, FP16_HALF, FP16_NEG_HALF, FP16_ZERO, FP16_ONE]
    k = [
        [FP16_ONE, FP16_NEG_ONE, FP16_ONE, FP16_NEG_ONE, FP16_HALF, FP16_HALF, FP16_ONE, FP16_ZERO],
        [FP16_NEG_ONE, FP16_ONE, FP16_NEG_ONE, FP16_ONE, FP16_NEG_HALF, FP16_HALF, FP16_ZERO, FP16_ONE],
    ]
    v = [list(reversed(row)) for row in k]
    trace = paper_single_head_attention_bit_model(q, k, v)
    assert len(trace.raw_scores) == 2
    assert len(trace.probabilities) == 2
    assert len(trace.output) == len(q)
    assert trace.array_counters["cell_count"] == 128


def test_paper_attention_cycle_model_is_structural():
    estimate = estimate_paper_attention_cycles(8, 8)
    assert estimate.qk_cycles > 0
    assert estimate.softmax_cycles > 0
    assert estimate.sv_cycles > 0
    assert estimate.total_cycles == (
        estimate.qk_cycles
        + estimate.softmax_cycles
        + estimate.sv_cycles
        + estimate.mode_switch_cycles
    )
