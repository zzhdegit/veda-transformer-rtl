from fractions import Fraction
from math import sqrt

from model.arithmetic.fp16_fp32_reference import fp16_to_fp32_bits
from model.arithmetic.fp32_mac_reference import fp32_bits_to_fraction, fraction_to_fp32_bits
from model.projection.fp32_fp16_reference import fp32_to_fp16_bits
from model.projection.projection_mha_reference import ProjectionMhaReference
from model.projection.projection_reference import WQ, WK, WO, WV, output_projection


FP16_ZERO = 0x0000
FP16_ONE = 0x3C00
FP16_HALF = 0x3800
FP16_NEG_HALF = 0xB800
FP16_NEG_ONE = 0xBC00
FP16_TWO = 0x4000
FP16_QUARTER = 0x3400
FP16_NEG_QUARTER = 0xB400


def fp32(value):
    return fraction_to_fp32_bits(value)


def identity(d_model):
    return [[FP16_ONE if out_idx == in_idx else FP16_ZERO for in_idx in range(d_model)] for out_idx in range(d_model)]


def zero(d_model):
    return [[FP16_ZERO for _ in range(d_model)] for _ in range(d_model)]


def diagonal(d_model):
    scale = [FP16_ONE, FP16_HALF, FP16_NEG_HALF, FP16_NEG_ONE]
    return [[scale[out_idx % len(scale)] if out_idx == in_idx else FP16_ZERO for in_idx in range(d_model)] for out_idx in range(d_model)]


def permutation(d_model):
    return [[FP16_ONE if in_idx == (out_idx * 3 + 1) % d_model else FP16_ZERO for in_idx in range(d_model)] for out_idx in range(d_model)]


def dense_pattern(d_model):
    pool = [FP16_QUARTER, FP16_NEG_QUARTER, FP16_HALF, FP16_NEG_HALF, FP16_ZERO, FP16_ONE, FP16_NEG_ONE]
    rows = []
    for out_idx in range(d_model):
        row = []
        for in_idx in range(d_model):
            row.append(pool[(out_idx * 5 + in_idx * 3 + 1) % len(pool)])
        rows.append(row)
    return rows


def head_outputs(n_head, d_head):
    values = []
    for head in range(n_head):
        row = []
        for dim in range(d_head):
            sign = -1 if ((head + dim) & 1) else 1
            row.append(fp32(Fraction(sign * (head + 1) * (dim + 1), 32)))
        values.append(row)
    return values


def vector_stats(bit_model, high_precision):
    diffs = []
    dot = Fraction(0, 1)
    norm_a = Fraction(0, 1)
    norm_b = Fraction(0, 1)
    for got_bits, ref_value in zip(bit_model, high_precision):
        got = fp32_bits_to_fraction(got_bits)
        diff = got - ref_value
        diffs.append(diff)
        dot += got * ref_value
        norm_a += got * got
        norm_b += ref_value * ref_value
    abs_diffs = [abs(value) for value in diffs]
    mse = sum(float(value * value) for value in diffs) / len(diffs)
    mae = sum(float(value) for value in abs_diffs) / len(abs_diffs)
    ref_l2 = sqrt(float(norm_b)) if norm_b else 0.0
    rel_l2 = sqrt(sum(float(value * value) for value in diffs)) / (ref_l2 + 1.0e-30)
    cosine = float(dot) / ((sqrt(float(norm_a)) * sqrt(float(norm_b))) + 1.0e-30)
    return {
        "max_abs": max(float(value) for value in abs_diffs),
        "mae": mae,
        "rmse": sqrt(mse),
        "relative_l2": rel_l2,
        "cosine": cosine,
    }


def high_precision_wo(heads, weights, n_head, d_head):
    concat = []
    for head in range(n_head):
        concat.extend(fp32_bits_to_fraction(value) for value in heads[head])
    quantized = [fp32_to_fp16_bits(fraction_to_fp32_bits(value)).output_bits for value in concat]
    inputs = [fp32_bits_to_fraction(fp16_to_fp32_bits(value)["output_bits"]) for value in quantized]
    outputs = []
    for row in weights:
        acc = Fraction(0, 1)
        for lhs, rhs_bits in zip(inputs, row):
            acc += lhs * fp32_bits_to_fraction(fp16_to_fp32_bits(rhs_bits)["output_bits"])
        outputs.append(acc)
    return outputs


def test_output_projection_cases():
    for d_model in (8, 16, 32):
        n_head = 1 if d_model == 8 else 2
        d_head = d_model // n_head
        heads = head_outputs(n_head, d_head)
        for weights in (identity(d_model), zero(d_model), diagonal(d_model), permutation(d_model), dense_pattern(d_model)):
            trace = output_projection(heads, weights, n_head, d_head, pe_num=8)
            assert trace.concat_fp32 == [value for row in heads for value in row]
            assert trace.concat_fp16 == [fp32_to_fp16_bits(value).output_bits for row in heads for value in row]
            high = high_precision_wo(heads, weights, n_head, d_head)
            stats = vector_stats(trace.output_fp32, high)
            assert stats["cosine"] <= 1.0000001
            assert len(trace.output_fp32) == d_model


def test_projection_mha_trace_nodes_and_cache_full():
    n_head = 2
    d_head = 8
    d_model = n_head * d_head
    weights = {WQ: identity(d_model), WK: identity(d_model), WV: identity(d_model), WO: dense_pattern(d_model)}
    ref = ProjectionMhaReference(n_head, d_head, max_seq_len=2, pe_num=8, weights=weights)
    hidden = [FP16_ONE, FP16_HALF, FP16_NEG_HALF, FP16_ZERO] * 4
    first = ref.run_token(hidden, meta=0x6E01)
    second = ref.run_token(hidden, meta=0x6E02)
    full = ref.run_token(hidden, meta=0x6E03)
    for trace in (first, second):
        assert not trace.invalid
        assert len(trace.head_output_fp32) == n_head
        assert len(trace.concat_fp32) == d_model
        assert len(trace.concat_fp16) == d_model
        assert trace.wo_output_fp32 == trace.final_output_fp32
    assert full.invalid
    assert full.concat_fp16 == []
    assert full.wo_output_fp32 == []
