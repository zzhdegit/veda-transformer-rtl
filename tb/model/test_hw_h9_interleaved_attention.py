from model.attention.paper_interleaved_attention_reference import (
    h9_native_inner_product,
    h9_paper_interleaved_attention_bit_model,
)
from model.attention.paper_interleaved_compare_h8 import compare_h9_to_h8
from model.attention.paper_interleaved_cycle_model import estimate_h9_interleaved_cycles
from model.pe_array.paper_array_mapping import h9_native_cell_index_for_dim


FP16_ZERO = 0x0000
FP16_ONE = 0x3C00
FP16_NEG_ONE = 0xBC00
FP16_TWO = 0x4000
FP16_HALF = 0x3800
FP16_NEG_HALF = 0xB800


def deterministic_values(length):
    pool = [FP16_ZERO, FP16_ONE, FP16_NEG_ONE, FP16_TWO, FP16_HALF, FP16_NEG_HALF, 0x3400, 0xB400, 0x3000]
    return [pool[(index * 5 + 3) % len(pool)] for index in range(length)]


def make_attention_case(d_head, seq_len):
    q = deterministic_values(d_head)
    k = []
    v = []
    for token in range(seq_len):
        row = deterministic_values(d_head)
        if token % 2:
            row = list(reversed(row))
        if token % 3 == 2:
            row = row[token % d_head :] + row[: token % d_head]
        k.append(row)
        v.append(list(reversed(row)) if token % 2 else row)
    return q, k, v


def test_h9_native_mapping_is_not_low_eight_only():
    indices = [h9_native_cell_index_for_dim(dim) for dim in range(8)]
    assert indices != list(range(8))
    assert len(set(indices)) == 8
    assert any(index >= 64 for index in indices)
    assert max(indices) >= 56


def test_h9_native_mapping_uses_both_groups_and_rows_for_small_heads():
    q, k, _ = make_attention_case(16, 1)
    result = h9_native_inner_product(q, k[0])
    active = result["active_counter"]
    groups = {group for group, _, _ in active}
    rows = {row for _, row, _ in active}
    assert groups == {0, 1}
    assert len(rows) == 8


def test_h9_attention_packets_and_activity_for_required_dimensions():
    for d_head in (8, 16, 64, 128):
        q, k, v = make_attention_case(d_head, 7)
        trace = h9_paper_interleaved_attention_bit_model(q, k, v)
        assert len(trace.score_packets) == 7
        assert len(trace.probability_packets) == 7
        assert [packet.score_index for packet in trace.score_packets] == list(range(7))
        assert [packet.probability_index for packet in trace.probability_packets] == list(range(7))
        assert trace.group_active_counts.get(0, 0) > 0
        assert trace.group_active_counts.get(1, 0) > 0
        assert len(trace.output) == d_head
        assert any(count > 0 for count in trace.column_active_counts.values())


def test_h9_attention_irregular_sequence_lengths_and_tails():
    for seq_len in (1, 2, 3, 7, 8, 9, 15, 16, 31, 32):
        q, k, v = make_attention_case(16, seq_len)
        trace = h9_paper_interleaved_attention_bit_model(q, k, v)
        assert len(trace.raw_scores) == seq_len
        assert trace.score_packets[-1].last_in_head is True
        assert trace.probability_packets[-1].last_probability is True


def test_h9_vs_h8_reports_metrics_when_add_order_differs():
    q, k, v = make_attention_case(64, 8)
    comparison = compare_h9_to_h8(q, k, v)
    assert "max_abs_error" in comparison["metrics"]["output"]
    assert "mae" in comparison["metrics"]["output"]
    assert "rmse" in comparison["metrics"]["output"]
    assert "relative_l2" in comparison["metrics"]["output"]
    assert "cosine_similarity" in comparison["metrics"]["output"]
    assert "max_ulp" in comparison["metrics"]["output"]


def test_h9_cycle_model_has_required_overlaps_and_speedup():
    for seq_len in (8, 16, 32):
        estimate = estimate_h9_interleaved_cycles(64, seq_len)
        assert estimate.qk_sfu_overlap_cycles > 0
        assert estimate.sfu_sv_overlap_cycles > 0
        assert estimate.interleaved_cycles < estimate.full_array_non_interleaved_cycles
        assert estimate.score_fifo_peak_occupancy > 0
        assert estimate.probability_fifo_peak_occupancy > 0
        assert estimate.array_utilization > 0.0
        assert estimate.sfu_utilization > 0.0


def test_h9_cycle_model_matches_rtl_calibration_points():
    staged_expected = {
        8: {1: 91, 2: 187, 8: 691, 16: 1363, 32: 2707, 64: 5395},
        16: {1: 165, 2: 330, 8: 1248, 16: 2472, 32: 4920, 64: 9816},
        64: {1: 609, 2: 1188, 8: 4590, 16: 9126, 32: 18198, 64: 36342},
    }
    interleaved_expected = {
        8: {1: 194, 2: 259, 8: 649, 16: 1169, 32: 2209, 64: 4289},
        16: {1: 196, 2: 261, 8: 651, 16: 1171, 32: 2211, 64: 4291},
        64: {1: 208, 2: 273, 8: 663, 16: 1183, 32: 2223, 64: 4303},
    }
    for d_head in (8, 16, 64):
        for seq_len in (1, 2, 8, 16, 32, 64):
            estimate = estimate_h9_interleaved_cycles(d_head, seq_len)
            assert estimate.staged_h8_cycles == staged_expected[d_head][seq_len]
            assert estimate.interleaved_cycles == interleaved_expected[d_head][seq_len]
