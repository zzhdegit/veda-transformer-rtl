from model.arithmetic.fp16_fp32_reference import fp16_to_fp32_bits
from model.arithmetic.fp32_add_reference import fp32_add
from model.arithmetic.fp32_mac_reference import fp32_mac
from model.pe.pe_core_reference import MODE_GEMV, MODE_QK_INNER, active_mask_for_width, inner_product_tiles, outer_product_sequence
from model.pe.pe_lane_reference import PE_LANE_MODE_FMA, PE_LANE_MODE_PRODUCT, pe_lane_compute
from model.pe.reduction_tree_reference import balanced_reduction


FP16_ONE = 0x3C00
FP16_TWO = 0x4000
FP16_HALF = 0x3800
FP16_NEG_ONE = 0xBC00
FP16_NEG_HALF = 0xB800
FP16_ZERO = 0x0000
FP16_NEG_ZERO = 0x8000


def f16(bits):
    return fp16_to_fp32_bits(bits)["output_bits"]


def test_fp32_add_directed_bit_results():
    assert fp32_add(0x3F800000, 0x40000000).output_bits == 0x40400000
    assert fp32_add(0x3F800000, 0xBF800000).output_bits == 0x00000000
    assert fp32_add(0x80000000, 0x00000000).output_bits == 0x00000000
    assert fp32_add(0x7F800000, 0x00000000).invalid


def test_pe_lane_product_fma_and_mask():
    assert pe_lane_compute(PE_LANE_MODE_PRODUCT, f16(FP16_TWO), f16(FP16_HALF)).output_bits == 0x3F800000
    assert pe_lane_compute(PE_LANE_MODE_FMA, 0x3F000000, f16(FP16_TWO), 0x3F800000).output_bits == 0x40000000
    assert pe_lane_compute(PE_LANE_MODE_PRODUCT, 0x7F800000, f16(FP16_ONE), lane_active=False).output_bits == 0
    assert pe_lane_compute(PE_LANE_MODE_FMA, 0x7F800000, f16(FP16_ONE), 0x3F800000, lane_active=False).output_bits == 0x3F800000


def test_reduction_tree_fixed_order_and_masks():
    values = [f16(x) for x in (FP16_ONE, FP16_TWO, FP16_NEG_ONE, FP16_HALF, FP16_NEG_HALF, FP16_ZERO, FP16_ONE, FP16_NEG_ZERO)]
    sum8, invalid, levels = balanced_reduction(values, 0xFF, trace=True)
    assert not invalid
    assert levels[0] == values
    assert len(levels) == 4
    assert sum8 == 0x40400000

    masked5, invalid5 = balanced_reduction(values, active_mask_for_width(5, 8))
    assert not invalid5
    assert masked5 == 0x40000000

    for pe_num, active in ((2, 1), (4, 3), (8, 5)):
        vals = [f16(FP16_ONE if idx % 2 == 0 else FP16_NEG_HALF) for idx in range(pe_num)]
        got, got_invalid = balanced_reduction(vals, active_mask_for_width(active, pe_num))
        assert not got_invalid
        assert 0 <= got <= 0xFFFFFFFF


def test_inner_product_tiling_boundaries():
    pe_num = 8
    dims = [1, pe_num - 1, pe_num, pe_num + 1, 2 * pe_num - 1, 2 * pe_num, 13, 128]
    pattern_q = [FP16_ONE, FP16_HALF, FP16_NEG_ONE, FP16_TWO]
    pattern_k = [FP16_TWO, FP16_NEG_ONE, FP16_HALF, FP16_ONE]
    for dim in dims:
        q = [pattern_q[idx % len(pattern_q)] for idx in range(dim)]
        k = [pattern_k[idx % len(pattern_k)] for idx in range(dim)]
        score, tiles = inner_product_tiles(q, k, pe_num, MODE_QK_INNER)
        assert len(tiles) == (dim + pe_num - 1) // pe_num
        assert tiles[-1]["mask"] == active_mask_for_width(((dim - 1) % pe_num) + 1, pe_num)
        assert score == tiles[-1]["acc"]


def test_gemv_alias_uses_inner_path():
    q = [FP16_ONE, FP16_TWO, FP16_NEG_ONE, FP16_HALF, FP16_ONE]
    w = [FP16_HALF, FP16_ONE, FP16_NEG_ONE, FP16_TWO, FP16_NEG_HALF]
    gemv_score, _ = inner_product_tiles(q, w, 8, MODE_GEMV)
    qk_score, _ = inner_product_tiles(q, w, 8, MODE_QK_INNER)
    assert gemv_score == qk_score


def test_outer_product_feedback_and_lane_mask():
    probs = [0x3F800000, 0x3F000000, 0xBF000000]
    rows = [
        [FP16_ONE, FP16_TWO, FP16_NEG_ONE, FP16_HALF, FP16_ONE, FP16_ZERO, FP16_ONE, FP16_NEG_HALF],
        [FP16_TWO, FP16_ONE, FP16_ONE, FP16_NEG_ONE, FP16_HALF, FP16_ONE, FP16_ZERO, FP16_TWO],
        [FP16_ONE, FP16_ONE, FP16_TWO, FP16_ONE, FP16_ZERO, FP16_ONE, FP16_HALF, FP16_ONE],
    ]
    masks = [0xFF, 0x1F, 0x07]
    acc, steps = outer_product_sequence(probs, rows, 8, masks)
    assert len(steps) == 3
    manual_lane0 = 0
    for scalar, row in zip(probs, rows):
        manual_lane0 = fp32_mac(scalar, f16(row[0]), manual_lane0, "fused").output_bits
    assert acc[0] == manual_lane0
    assert acc[7] == steps[0]["acc"][7]
