from fractions import Fraction

from model.arithmetic.fp32_mac_reference import fp32_bits_to_fraction, fp32_mac, fraction_to_fp32_bits
from model.projection.projection_reference import WQ, WK, WO, WV
from model.transformer.ffn_reference import ffn_forward, output_row_major_address
from model.transformer.residual_reference import residual_add
from model.transformer.relu_reference import relu_fp32, relu_quantize
from model.transformer.rmsnorm_reference import EPS_FP32_DEFAULT, FP32_ZERO, mean_scale_for_d_model, rmsnorm
from model.transformer.transformer_layer_reference import TransformerLayerReference


FP16_ZERO = 0x0000
FP16_ONE = 0x3C00
FP16_HALF = 0x3800
FP16_NEG_HALF = 0xB800
FP16_NEG_ONE = 0xBC00
FP16_TWO = 0x4000
FP16_QUARTER = 0x3400
FP16_NEG_QUARTER = 0xB400

FP32_ONE = 0x3F800000
FP32_TWO = 0x40000000
FP32_HALF = 0x3F000000
FP32_NEG_ONE = 0xBF800000
FP32_NEG_ZERO = 0x80000000
FP32_INF = 0x7F800000
FP32_NAN = 0x7FC00001


def fp32_frac(value):
    return fraction_to_fp32_bits(value)


def identity(rows, cols):
    return [[FP16_ONE if out_idx == in_idx else FP16_ZERO for in_idx in range(cols)] for out_idx in range(rows)]


def zero(rows, cols):
    return [[FP16_ZERO for _ in range(cols)] for _ in range(rows)]


def dense(rows, cols):
    pool = [FP16_QUARTER, FP16_NEG_QUARTER, FP16_HALF, FP16_NEG_HALF, FP16_ZERO]
    return [[pool[(out_idx * 3 + in_idx * 5 + 1) % len(pool)] for in_idx in range(cols)] for out_idx in range(rows)]


def gamma_pattern(d_model):
    pool = [FP16_ONE, FP16_HALF, FP16_NEG_HALF, FP16_TWO]
    return [pool[idx % len(pool)] for idx in range(d_model)]


def hidden_pattern(d_model):
    pool = [FP16_ONE, FP16_HALF, FP16_NEG_HALF, FP16_ZERO, FP16_QUARTER, FP16_NEG_QUARTER]
    return [pool[(idx * 5 + 1) % len(pool)] for idx in range(d_model)]


def mha_weights(d_model, dense_weights=False):
    if dense_weights:
        base = dense(d_model, d_model)
        return {WQ: base, WK: dense(d_model, d_model), WV: dense(d_model, d_model), WO: dense(d_model, d_model)}
    return {
        WQ: identity(d_model, d_model),
        WK: identity(d_model, d_model),
        WV: identity(d_model, d_model),
        WO: zero(d_model, d_model),
    }


def test_rmsnorm_zero_constant_and_gamma_layout():
    for d_model in (8, 16, 32):
        trace = rmsnorm([FP32_ZERO for _ in range(d_model)], [FP16_ONE for _ in range(d_model)])
        assert not trace.invalid
        assert trace.sum_sq == FP32_ZERO
        assert trace.mean_scale == mean_scale_for_d_model(d_model)
        assert fp32_bits_to_fraction(trace.mean_scale) == Fraction(1, d_model)
        assert trace.mean_sq_eps == EPS_FP32_DEFAULT
        assert trace.norm_fp16 == [FP16_ZERO for _ in range(d_model)]

        constant = rmsnorm([FP32_ONE for _ in range(d_model)], gamma_pattern(d_model))
        assert not constant.invalid
        assert fp32_bits_to_fraction(constant.sum_sq) == Fraction(d_model, 1)
        assert len(constant.norm_fp32) == d_model
        assert len(constant.norm_fp16) == d_model
        assert constant.gamma_fp32[0] == FP32_ONE
        assert constant.gamma_fp32[1] == FP32_HALF
        assert constant.gamma_fp32[2] == 0xBF000000


def test_rmsnorm_uses_dimension_order_fused_accumulation_and_flags_nonfinite():
    values = [FP32_ONE, FP32_TWO, FP32_HALF, FP32_NEG_ONE, FP32_ZERO, FP32_ONE, FP32_HALF, FP32_TWO]
    trace = rmsnorm(values, [FP16_ONE for _ in values])
    acc = FP32_ZERO
    for idx, value in enumerate(values):
        acc = fp32_mac(value, value, acc, "fused").output_bits
        assert trace.sum_trace[idx]["acc"] == acc
    assert trace.sum_sq == acc

    bad = rmsnorm([FP32_INF] + [FP32_ZERO for _ in range(7)], [FP16_ONE for _ in range(8)])
    assert bad.invalid
    bad_gamma = rmsnorm([FP32_ONE for _ in range(8)], [0x7C00] + [FP16_ONE for _ in range(7)])
    assert bad_gamma.invalid


def test_residual_and_relu_boundaries():
    res = residual_add([FP32_ONE, FP32_NEG_ZERO], [FP32_NEG_ONE, FP32_ZERO])
    assert not res.invalid
    assert res.output_fp32 == [FP32_ZERO, FP32_ZERO]

    assert relu_fp32(FP32_NEG_ONE).output_bits == FP32_ZERO
    assert relu_fp32(FP32_NEG_ZERO).output_bits == FP32_ZERO
    assert relu_fp32(FP32_TWO).output_bits == FP32_TWO
    assert relu_fp32(FP32_INF).invalid
    assert relu_fp32(FP32_NAN).invalid

    relu = relu_quantize([FP32_NEG_ONE, FP32_NEG_ZERO, FP32_TWO, FP32_INF])
    assert relu.relu_fp32[0] == FP32_ZERO
    assert relu.relu_fp32[1] == FP32_ZERO
    assert relu.relu_fp32[2] == FP32_TWO
    assert relu.activation_fp16[2] == FP16_TWO
    assert relu.invalid


def test_ffn_shapes_and_row_major_addresses():
    for d_model in (8, 16, 32):
        d_ffn = 4 * d_model
        norm2 = hidden_pattern(d_model)
        w1 = identity(d_ffn, d_model)
        w2 = identity(d_model, d_ffn)
        trace = ffn_forward(norm2, w1, w2, pe_num=8)
        assert not trace.invalid
        assert len(trace.ffn1_fp32) == d_ffn
        assert len(trace.activation_fp16) == d_ffn
        assert len(trace.ffn2_fp32) == d_model
        assert output_row_major_address(3, 2, d_model) == 3 * d_model + 2
        assert output_row_major_address(3, 2, d_ffn) == 3 * d_ffn + 2


def test_transformer_layer_trace_nodes_and_cache_full():
    n_head = 2
    d_head = 8
    d_model = n_head * d_head
    d_ffn = 4 * d_model
    layer = TransformerLayerReference(
        n_head,
        d_head,
        max_seq_len=2,
        pe_num=8,
        mha_weights=mha_weights(d_model),
        gamma1_fp16=[FP16_ONE for _ in range(d_model)],
        gamma2_fp16=[FP16_ONE for _ in range(d_model)],
        w1=zero(d_ffn, d_model),
        w2=zero(d_model, d_ffn),
    )
    hidden = hidden_pattern(d_model)
    first = layer.run_token(hidden, meta=0x7101)
    second = layer.run_token(hidden, meta=0x7102)
    full = layer.run_token(hidden, meta=0x7103)

    for trace in (first, second):
        assert not trace.invalid
        assert len(trace.input_fp32) == d_model
        assert len(trace.norm1_fp16) == d_model
        assert len(trace.mha_fp32) == d_model
        assert len(trace.residual1_fp32) == d_model
        assert len(trace.norm2_fp16) == d_model
        assert len(trace.ffn1_fp32) == d_ffn
        assert len(trace.activation_fp16) == d_ffn
        assert len(trace.ffn2_fp32) == d_model
        assert len(trace.final_fp32) == d_model
        assert trace.final_fp32 == trace.residual1_fp32

    assert full.invalid
    assert full.final_fp32 == []


def test_transformer_layer_dense_h2d8_runs_multitoken():
    n_head = 2
    d_head = 8
    d_model = n_head * d_head
    d_ffn = 4 * d_model
    layer = TransformerLayerReference(
        n_head,
        d_head,
        max_seq_len=3,
        pe_num=8,
        mha_weights=mha_weights(d_model, dense_weights=True),
        gamma1_fp16=gamma_pattern(d_model),
        gamma2_fp16=list(reversed(gamma_pattern(d_model))),
        w1=dense(d_ffn, d_model),
        w2=dense(d_model, d_ffn),
    )
    traces = [layer.run_token(hidden_pattern(d_model), meta=0x7200 + idx) for idx in range(3)]
    assert all(not trace.invalid for trace in traces)
    assert all(len(trace.final_fp32) == d_model for trace in traces)
    assert traces[0].norm1_sum_sq != FP32_ZERO
    assert traces[0].norm2_inv_rms != FP32_ZERO
