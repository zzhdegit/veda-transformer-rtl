from fractions import Fraction

from model.arithmetic.fp16_fp32_reference import fp16_to_fp32_bits
from model.arithmetic.fp32_mac_reference import fraction_to_fp32_bits
from model.projection.fp32_fp16_reference import fp32_to_fp16_bits
from model.projection.gemv_reference import gemv, output_row_major_address
from model.projection.projection_mha_reference import ProjectionMhaReference
from model.projection.projection_reference import (
    WK,
    WO,
    WQ,
    WV,
    concat_attention_heads,
    output_projection,
    projection_output_index,
    qkv_projection,
)


FP16_ZERO = 0x0000
FP16_NEG_ZERO = 0x8000
FP16_ONE = 0x3C00
FP16_TWO = 0x4000
FP16_HALF = 0x3800
FP16_NEG_ONE = 0xBC00


def fp32(value):
    return fraction_to_fp32_bits(value)


def identity(d_model):
    rows = []
    for out_idx in range(d_model):
        row = []
        for in_idx in range(d_model):
            row.append(FP16_ONE if out_idx == in_idx else FP16_ZERO)
        rows.append(row)
    return rows


def diagonal(d_model, scale):
    rows = []
    for out_idx in range(d_model):
        row = []
        for in_idx in range(d_model):
            row.append(scale if out_idx == in_idx else FP16_ZERO)
        rows.append(row)
    return rows


def patterned_vector(d_model):
    pattern = [FP16_ONE, FP16_HALF, FP16_NEG_ONE, FP16_TWO, FP16_NEG_ZERO, FP16_ZERO]
    return [pattern[idx % len(pattern)] for idx in range(d_model)]


def test_fp32_to_fp16_special_policy():
    assert fp32_to_fp16_bits(0x00000000).output_bits == 0x0000
    assert fp32_to_fp16_bits(0x80000000).output_bits == 0x8000

    pos_sub = fp32_to_fp16_bits(0x00000001)
    assert pos_sub.output_bits == 0x0000
    assert pos_sub.underflow_or_ftz

    neg_sub = fp32_to_fp16_bits(0x80000001)
    assert neg_sub.output_bits == 0x8000
    assert neg_sub.underflow_or_ftz

    inf = fp32_to_fp16_bits(0x7F800000)
    assert inf.output_bits == 0x0000
    assert inf.invalid

    ninf = fp32_to_fp16_bits(0xFF800000)
    assert ninf.output_bits == 0x8000
    assert ninf.invalid

    nan = fp32_to_fp16_bits(0x7FC01234)
    assert nan.output_bits == 0x0000
    assert nan.invalid


def test_fp32_to_fp16_rne_tie_even_boundaries():
    one = fp32_to_fp16_bits(fp32(Fraction(1, 1)))
    assert one.output_bits == 0x3C00
    assert not one.inexact

    tie_to_even_lower = fp32_to_fp16_bits(fp32(Fraction(1, 1) + Fraction(1, 2048)))
    assert tie_to_even_lower.output_bits == 0x3C00
    assert tie_to_even_lower.inexact

    just_above_tie = fp32_to_fp16_bits(fp32(Fraction(1, 1) + Fraction(1, 2048) + Fraction(1, 1 << 23)))
    assert just_above_tie.output_bits == 0x3C01
    assert just_above_tie.inexact

    odd_lower_tie = fp32_to_fp16_bits(fp32(Fraction(1, 1) + Fraction(3, 2048)))
    assert odd_lower_tie.output_bits == 0x3C02
    assert odd_lower_tie.inexact


def test_fp32_to_fp16_overflow_underflow_and_carry():
    max_half = fp32_to_fp16_bits(fp32(Fraction(65504, 1)))
    assert max_half.output_bits == 0x7BFF
    assert not max_half.overflow

    overflow = fp32_to_fp16_bits(fp32(Fraction(70000, 1)))
    assert overflow.output_bits == 0x7BFF
    assert overflow.overflow
    assert overflow.inexact

    underflow = fp32_to_fp16_bits(fp32(Fraction(1, 1 << 15)))
    assert underflow.output_bits == 0x0000
    assert underflow.underflow_or_ftz

    rounded_exp_carry = fp32_to_fp16_bits(fp32(Fraction(4095, 2048)))
    assert rounded_exp_carry.output_bits == 0x4000


def test_mapping_and_weight_addresses():
    for n_head, d_head in [(1, 8), (2, 8), (4, 8), (2, 16)]:
        d_model = n_head * d_head
        for head in range(n_head):
            for dim in range(d_head):
                assert projection_output_index(head, dim, d_head) == head * d_head + dim
        assert output_row_major_address(d_model - 1, d_model - 1, d_model) == d_model * d_model - 1


def test_gemv_identity_and_tiling_layouts():
    for d_model in [8, 16, 32]:
        hidden = [value if value != FP16_NEG_ZERO else FP16_ZERO for value in patterned_vector(d_model)]
        trace = gemv(hidden, identity(d_model), pe_num=8)
        expected = [fp16_to_fp32_bits(value)["output_bits"] for value in hidden]
        assert trace.outputs == expected
        for row_idx, row_trace in enumerate(trace.rows):
            assert row_trace.output_index == row_idx
            assert row_trace.tiles[0]["base"] == 0


def test_qkv_projection_order_and_head_split():
    n_head = 2
    d_head = 8
    d_model = n_head * d_head
    hidden = patterned_vector(d_model)
    weights = {
        WQ: identity(d_model),
        WK: diagonal(d_model, FP16_TWO),
        WV: diagonal(d_model, FP16_HALF),
        WO: identity(d_model),
    }
    trace = qkv_projection(hidden, weights, n_head, d_head, pe_num=8)
    assert len(trace.q_fp32) == d_model
    assert len(trace.k_fp16) == d_model
    assert len(trace.v_fp16) == d_model
    for item_idx, item in enumerate(trace.qkv_stream):
        head = item_idx // d_head
        dim = item_idx % d_head
        assert item["head"] == head
        assert item["dim"] == dim
        assert item["last_dim"] == (dim == d_head - 1)
        assert item["last_head"] == (item_idx == d_model - 1)
        assert item["q_fp16"] == trace.q_heads[head][dim]
        assert item["k_fp16"] == trace.k_heads[head][dim]
        assert item["v_fp16"] == trace.v_heads[head][dim]


def test_concat_and_output_projection_quantization_order():
    n_head = 4
    d_head = 8
    d_model = n_head * d_head
    heads = []
    for head in range(n_head):
        row = []
        for dim in range(d_head):
            row.append(fp32(Fraction((head + 1) * (dim + 1), 16)))
        heads.append(row)
    concat = concat_attention_heads(heads, n_head, d_head)
    for head in range(n_head):
        for dim in range(d_head):
            assert concat[head * d_head + dim] == heads[head][dim]
    trace = output_projection(heads, identity(d_model), n_head, d_head, pe_num=8)
    assert trace.concat_fp32 == concat
    assert trace.concat_fp16 == [fp32_to_fp16_bits(value).output_bits for value in concat]
    assert trace.output_fp32 == [fp16_to_fp32_bits(value)["output_bits"] for value in trace.concat_fp16]


def test_projection_mha_reference_framework_cache_full():
    n_head = 1
    d_head = 8
    d_model = 8
    weights = {WQ: identity(d_model), WK: identity(d_model), WV: identity(d_model), WO: identity(d_model)}
    ref = ProjectionMhaReference(n_head, d_head, max_seq_len=2, pe_num=8, weights=weights)
    hidden = [FP16_ONE] * d_model
    first = ref.run_token(hidden, meta=0x601)
    second = ref.run_token(hidden, meta=0x602)
    full = ref.run_token(hidden, meta=0x603)
    assert not first.invalid
    assert not second.invalid
    assert full.invalid
    assert full.final_output_fp32 == []
