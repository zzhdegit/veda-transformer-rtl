from model.attention.softmax_reference import fp32_to_float
from model.pe_array.paper_array_8x8x2_reference import PaperArray8x8x2Reference
from model.pe_array.paper_array_compare_legacy import compare_inner, compare_outer
from model.pe_array.paper_array_mapping import (
    ARRAY_COLS,
    ARRAY_GROUPS,
    ARRAY_ROWS,
    MODE_INNER_PRODUCT,
    MODE_OUTER_PRODUCT,
    PE_CELLS,
)
from model.pe_array.paper_pe_reference import expected_type_a_columns, expected_type_b_columns


FP16_ZERO = 0x0000
FP16_NEG_ZERO = 0x8000
FP16_ONE = 0x3C00
FP16_NEG_ONE = 0xBC00
FP16_TWO = 0x4000
FP16_HALF = 0x3800
FP16_QUARTER = 0x3400
FP16_MIN_NORMAL = 0x0400
FP16_MAX_NORMAL = 0x7BFF


def deterministic_values(length):
    pool = [FP16_ONE, FP16_NEG_ONE, FP16_TWO, FP16_HALF, FP16_QUARTER, 0xB800, 0x3000, 0xB400]
    return [pool[(index * 5 + 3) % len(pool)] for index in range(length)]


def test_array_has_128_cells_and_type_mapping():
    array = PaperArray8x8x2Reference()
    assert array.cell_count() == 128
    assert len(array.groups) == ARRAY_GROUPS
    for group in array.groups:
        assert len(group.cells) == ARRAY_ROWS * ARRAY_COLS
        for cell in group.cells:
            if cell.column in expected_type_a_columns():
                assert cell.pe_type == "A"
            if cell.column in expected_type_b_columns():
                assert cell.pe_type == "B"


def test_inner_all_zero_identity_and_powers_of_two():
    array = PaperArray8x8x2Reference()
    zero = array.inner_product([FP16_ZERO] * 8, [FP16_ZERO] * 8)
    assert zero.mode == MODE_INNER_PRODUCT
    assert zero.scalar == 0
    assert zero.active_cells == 8

    one = array.inner_product([FP16_ONE] + [FP16_ZERO] * 7, [FP16_ONE] + [FP16_ZERO] * 7)
    assert fp32_to_float(one.scalar) == 1.0

    powers = array.inner_product([FP16_ONE, FP16_TWO, FP16_HALF, FP16_QUARTER], [FP16_ONE, FP16_HALF, FP16_TWO, FP16_ONE])
    assert fp32_to_float(powers.scalar) == 3.25


def test_inner_dense_mixed_cancellation_and_groups():
    q = deterministic_values(128)
    k = list(reversed(deterministic_values(128)))
    both = PaperArray8x8x2Reference().inner_product(q, k, group_mask=0x3)
    group0 = PaperArray8x8x2Reference().inner_product(q, k, group_mask=0x1)
    group1 = PaperArray8x8x2Reference().inner_product(q, k, group_mask=0x2)
    assert both.active_cells == 128
    assert group0.active_cells == 64
    assert group1.active_cells == 64
    assert both.tile_traces[0]["groups"][0].active_cells == 64
    assert both.tile_traces[0]["groups"][1].active_cells == 64

    cancel = PaperArray8x8x2Reference().inner_product([FP16_ONE, FP16_ONE], [FP16_ONE, FP16_NEG_ONE])
    assert fp32_to_float(cancel.scalar) == 0.0


def test_outer_identity_mixed_and_group_masks():
    prob_one = [0x3F800000]
    row = [[FP16_ONE, FP16_TWO, FP16_NEG_ONE, FP16_HALF]]
    result = PaperArray8x8x2Reference().outer_product(prob_one, row)
    assert result.mode == MODE_OUTER_PRODUCT
    assert [fp32_to_float(value) for value in result.vector[:4]] == [1.0, 2.0, -1.0, 0.5]

    probs = [0x3F000000, 0x3F000000]
    rows = [[FP16_ONE, FP16_NEG_ONE], [FP16_ONE, FP16_ONE]]
    mixed = PaperArray8x8x2Reference().outer_product(probs, rows)
    assert [fp32_to_float(value) for value in mixed.vector[:2]] == [1.0, 0.0]

    one_group = PaperArray8x8x2Reference().outer_product(prob_one, [deterministic_values(80)], vector_length=80, group_mask=0x1)
    assert one_group.active_cells == 64
    assert all(value == 0 for value in one_group.vector[64:80])


def test_tail_masks_partial_rows_columns_and_multiple_tiles():
    q = deterministic_values(160)
    k = deterministic_values(160)
    result = PaperArray8x8x2Reference().inner_product(q, k)
    assert len(result.tile_traces) == 2
    assert result.active_cells == 160

    partial = PaperArray8x8x2Reference().inner_product(q[:31], k[:31], row_mask=0x0F, column_mask=0x7F)
    assert partial.active_cells == 28

    rows = [deterministic_values(160), list(reversed(deterministic_values(160)))]
    outer = PaperArray8x8x2Reference().outer_product([0x3F000000, 0x3F000000], rows, vector_length=160)
    assert len(outer.tile_traces) == 2
    assert len(outer.vector) == 160


def test_mode_switch_reset_and_repeated_command():
    array = PaperArray8x8x2Reference()
    q = deterministic_values(16)
    k = deterministic_values(16)
    first = array.inner_product(q, k)
    second = array.outer_product([0x3F800000], [q], vector_length=16)
    third = array.inner_product(q, k)
    assert first.mode_switch is False
    assert second.mode_switch is True
    assert third.mode_switch is True
    assert array.counters()["mode_switch_count"] == 2

    repeat = PaperArray8x8x2Reference()
    a = repeat.inner_product(q, k)
    b = repeat.inner_product(q, k)
    assert a.scalar == b.scalar

    array.reset()
    assert array.counters()["command_count"] == 0
    clean = array.inner_product([FP16_ONE], [FP16_ONE])
    assert clean.mode_switch is False
    assert fp32_to_float(clean.scalar) == 1.0


def test_signed_zero_min_max_normal_and_group1_only():
    zero = PaperArray8x8x2Reference().inner_product([FP16_NEG_ZERO, FP16_ONE], [FP16_ONE, FP16_ZERO])
    assert zero.scalar == 0

    normals = PaperArray8x8x2Reference().inner_product([FP16_MIN_NORMAL, FP16_ONE], [FP16_ONE, FP16_MIN_NORMAL])
    assert normals.active_cells == 2
    assert normals.invalid is False

    max_case = PaperArray8x8x2Reference().outer_product([0x3F800000], [[FP16_MAX_NORMAL]], vector_length=1)
    assert max_case.vector[0] != 0

    q = deterministic_values(80)
    k = deterministic_values(80)
    group1 = PaperArray8x8x2Reference().inner_product(q, k, group_mask=0x2)
    assert group1.active_cells == 16


def test_paper_vs_legacy_comparison_reports_metrics():
    q = deterministic_values(16)
    k = list(reversed(deterministic_values(16)))
    inner = compare_inner(q, k)
    assert "max_abs_error" in inner["metrics"]
    assert "max_ulp" in inner["metrics"]

    probabilities = [0x3F000000, 0x3F000000]
    rows = [q, k]
    outer = compare_outer(probabilities, rows)
    assert outer["bit_exact"] is True
    assert outer["metrics"]["argmax_match"] is True

