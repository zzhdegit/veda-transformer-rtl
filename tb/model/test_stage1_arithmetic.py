from __future__ import annotations

from model.arithmetic.arithmetic_reference import (
    compare_max,
    round_and_saturate,
    signed_add,
    signed_mac,
    signed_mul,
)
from model.arithmetic.numeric_format import bits_from_signed, signed_from_bits


def test_signed_mul_full_width_edges():
    cases = [
        (0, 7),
        (3, 4),
        (-3, 4),
        (-8, -1),
        (7, 7),
    ]
    for a, b in cases:
        got = signed_mul(bits_from_signed(a, 4), bits_from_signed(b, 4), 4, 4, 8)
        assert signed_from_bits(got.output_bits, 8) == a * b
        assert got.high_precision_result == a * b
        assert not got.overflow


def test_signed_mul_reports_truncation_overflow():
    got = signed_mul(bits_from_signed(7, 4), bits_from_signed(7, 4), 4, 4, 5)
    assert got.high_precision_result == 49
    assert signed_from_bits(got.output_bits, 5) == signed_from_bits(49, 5)
    assert got.overflow


def test_signed_add_with_extended_output():
    got = signed_add(bits_from_signed(127, 8), bits_from_signed(1, 8), 8, 8, 9)
    assert signed_from_bits(got.output_bits, 9) == 128
    assert not got.overflow


def test_signed_add_overflow_when_output_is_narrow():
    got = signed_add(bits_from_signed(127, 8), bits_from_signed(1, 8), 8, 8, 8)
    assert signed_from_bits(got.output_bits, 8) == -128
    assert got.overflow


def test_signed_mac_clear_and_accumulate():
    clear = signed_mac(
        bits_from_signed(3, 4),
        bits_from_signed(-5, 4),
        bits_from_signed(99, 12),
        4,
        4,
        12,
        clear=True,
    )
    assert signed_from_bits(clear.output_bits, 12) == -15

    accum = signed_mac(
        bits_from_signed(3, 4),
        bits_from_signed(-5, 4),
        bits_from_signed(20, 12),
        4,
        4,
        12,
        clear=False,
    )
    assert signed_from_bits(accum.output_bits, 12) == 5


def test_compare_max_tie_selects_a():
    out_bits, take_b = compare_max(bits_from_signed(-2, 8), bits_from_signed(-2, 8), 8)
    assert signed_from_bits(out_bits, 8) == -2
    assert not take_b

    out_bits, take_b = compare_max(bits_from_signed(-2, 8), bits_from_signed(5, 8), 8)
    assert signed_from_bits(out_bits, 8) == 5
    assert take_b


def test_round_sat_truncate_and_saturate():
    got = round_and_saturate(bits_from_signed(0x0800, 16), 16, 8, frac_drop=4, rounding="truncate")
    assert signed_from_bits(got.output_bits, 8) == 127
    assert got.overflow
    assert not got.underflow

    got = round_and_saturate(bits_from_signed(-0x0810, 16), 16, 8, frac_drop=4, rounding="truncate")
    assert signed_from_bits(got.output_bits, 8) == -128
    assert got.underflow


def test_round_sat_nearest_even_boundaries():
    # +1.5 rounds to +2, +2.5 ties to even and remains +2.
    got = round_and_saturate(bits_from_signed(3, 8), 8, 4, frac_drop=1, rounding="nearest_even")
    assert signed_from_bits(got.output_bits, 4) == 2
    assert got.inexact

    got = round_and_saturate(bits_from_signed(5, 8), 8, 4, frac_drop=1, rounding="nearest_even")
    assert signed_from_bits(got.output_bits, 4) == 2
    assert got.inexact

    got = round_and_saturate(bits_from_signed(-3, 8), 8, 4, frac_drop=1, rounding="nearest_even")
    assert signed_from_bits(got.output_bits, 4) == -2
    assert got.inexact

    got = round_and_saturate(bits_from_signed(-5, 8), 8, 4, frac_drop=1, rounding="nearest_even")
    assert signed_from_bits(got.output_bits, 4) == -2
    assert got.inexact

    got = round_and_saturate(bits_from_signed(-7, 8), 8, 4, frac_drop=1, rounding="nearest_even")
    assert signed_from_bits(got.output_bits, 4) == -4
    assert got.inexact
