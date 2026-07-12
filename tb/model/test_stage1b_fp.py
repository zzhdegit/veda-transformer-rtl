from fractions import Fraction

from model.arithmetic.fp16_fp32_reference import fp16_to_fp32_bits
from model.arithmetic.fp32_mac_reference import (
    find_fused_discriminator,
    fp32_bits_to_fraction,
    fp32_mac,
)


def test_fp16_to_fp32_directed_values():
    assert fp16_to_fp32_bits(0x0000)["output_bits"] == 0x00000000
    assert fp16_to_fp32_bits(0x8000)["output_bits"] == 0x80000000
    assert fp16_to_fp32_bits(0x0400)["output_bits"] == 0x38800000
    assert fp16_to_fp32_bits(0x7BFF)["output_bits"] == 0x477FE000
    assert fp16_to_fp32_bits(0x3E00)["output_bits"] == 0x3FC00000
    assert fp16_to_fp32_bits(0xBC00)["output_bits"] == 0xBF800000


def test_fp16_to_fp32_special_policy():
    sub = fp16_to_fp32_bits(0x0001)
    assert sub["output_bits"] == 0x00000000
    assert sub["underflow_or_ftz"]
    assert not sub["invalid"]

    neg_sub = fp16_to_fp32_bits(0x8001)
    assert neg_sub["output_bits"] == 0x80000000
    assert neg_sub["underflow_or_ftz"]

    for bits in (0x7C00, 0xFC00, 0x7E01):
        result = fp16_to_fp32_bits(bits)
        assert result["invalid"]
        assert result["output_bits"] in (0x00000000, 0x80000000)


def test_fp16_to_fp32_exhaustive_no_x_policy():
    invalid_count = 0
    ftz_count = 0
    for bits in range(65536):
        result = fp16_to_fp32_bits(bits)
        assert 0 <= result["output_bits"] <= 0xFFFFFFFF
        invalid_count += int(result["invalid"])
        ftz_count += int(result["underflow_or_ftz"])
    assert invalid_count == 2048
    assert ftz_count == 2046


def test_fp32_fraction_conversion_is_exact_for_simple_values():
    assert fp32_bits_to_fraction(0x3F800000) == 1
    assert fp32_bits_to_fraction(0xBF800000) == -1
    assert fp32_bits_to_fraction(0x3FC00000) == Fraction(3, 2)


def test_fp32_mac_directed_bit_results():
    assert fp32_mac(0x3FC00000, 0x40100000, 0x3F800000, "fused").output_bits == 0x408C0000
    assert fp32_mac(0x3FC00000, 0x40100000, 0x3F800000, "non_fused").output_bits == 0x408C0000
    assert fp32_mac(0x3F800001, 0x3F800000, 0xBF800000, "fused").output_bits == 0x34000000
    assert fp32_mac(0x00800000, 0x3F800000, 0x80800000, "fused").output_bits == 0x00000000


def test_fused_non_fused_discriminator_exists():
    a, b, c, fused, non_fused = find_fused_discriminator()
    assert fp32_mac(a, b, c, "fused").output_bits == fused
    assert fp32_mac(a, b, c, "non_fused").output_bits == non_fused
    assert fused != non_fused
