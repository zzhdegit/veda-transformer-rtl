"""Bit-accurate Stage 6 FP32-to-FP16 conversion reference."""

from collections import namedtuple
from fractions import Fraction

from model.arithmetic.fp32_mac_reference import (
    classify_fp32,
    fp32_bits_to_fraction,
    fraction_to_fp32_bits,
)


FP16_EXP_W = 5
FP16_FRAC_W = 10
FP16_EXP_BIAS = 15
FP16_MAX_EXP = 0x1E
FP16_MAX_FINITE = 0x7BFF
FP16_MIN_NORMAL_EXP = -14
FP16_MAX_NORMAL_EXP = 15

Fp32ToFp16Result = namedtuple(
    "Fp32ToFp16Result",
    [
        "input_bits",
        "output_bits",
        "invalid",
        "overflow",
        "underflow_or_ftz",
        "inexact",
        "category",
    ],
)


def _pow2_fraction(exp):
    if exp >= 0:
        return Fraction(1 << exp, 1)
    return Fraction(1, 1 << (-exp))


def _floor_log2_positive(value):
    if value <= 0:
        raise ValueError("value must be positive")
    numerator = value.numerator
    denominator = value.denominator
    exp = numerator.bit_length() - denominator.bit_length()
    while _pow2_fraction(exp) > value:
        exp -= 1
    while _pow2_fraction(exp + 1) <= value:
        exp += 1
    return exp


def _round_fraction_to_int_rne(value):
    if value < 0:
        raise ValueError("value must be non-negative")
    quotient, remainder = divmod(value.numerator, value.denominator)
    twice = remainder * 2
    if twice > value.denominator:
        return quotient + 1
    if twice < value.denominator:
        return quotient
    return quotient + (quotient & 1)


def _fp16_bits_to_fraction(bits):
    bits &= 0xFFFF
    sign = -1 if (bits >> 15) & 1 else 1
    exp = (bits >> FP16_FRAC_W) & 0x1F
    frac = bits & ((1 << FP16_FRAC_W) - 1)
    if exp == 0:
        return Fraction(0, 1)
    if exp == 0x1F:
        raise ValueError("non-finite FP16")
    significand = (1 << FP16_FRAC_W) | frac
    return sign * Fraction(significand, 1 << FP16_FRAC_W) * _pow2_fraction(exp - FP16_EXP_BIAS)


def fp32_to_fp16_bits(bits):
    bits &= 0xFFFFFFFF
    sign = (bits >> 31) & 1
    category = classify_fp32(bits)

    if category in ("inf", "nan"):
        return Fp32ToFp16Result(bits, sign << 15, True, False, False, True, category)

    if category == "zero":
        return Fp32ToFp16Result(bits, sign << 15, False, False, False, False, category)

    if category == "subnormal":
        return Fp32ToFp16Result(bits, sign << 15, False, False, True, True, category)

    value = fp32_bits_to_fraction(bits)
    magnitude = -value if value < 0 else value
    min_normal = _pow2_fraction(FP16_MIN_NORMAL_EXP)
    max_finite = _fp16_bits_to_fraction(FP16_MAX_FINITE)

    if magnitude < min_normal:
        return Fp32ToFp16Result(bits, sign << 15, False, False, True, True, category)

    if magnitude > max_finite:
        return Fp32ToFp16Result(
            bits,
            (sign << 15) | FP16_MAX_FINITE,
            False,
            True,
            False,
            True,
            category,
        )

    exponent = _floor_log2_positive(magnitude)
    scaled = magnitude / _pow2_fraction(exponent - FP16_FRAC_W)
    significand = _round_fraction_to_int_rne(scaled)
    if significand == (1 << (FP16_FRAC_W + 1)):
        significand >>= 1
        exponent += 1

    if exponent > FP16_MAX_NORMAL_EXP:
        return Fp32ToFp16Result(
            bits,
            (sign << 15) | FP16_MAX_FINITE,
            False,
            True,
            False,
            True,
            category,
        )

    if exponent < FP16_MIN_NORMAL_EXP:
        return Fp32ToFp16Result(bits, sign << 15, False, False, True, True, category)

    exp16 = exponent + FP16_EXP_BIAS
    frac16 = significand - (1 << FP16_FRAC_W)
    out = ((sign << 15) | (exp16 << FP16_FRAC_W) | (frac16 & ((1 << FP16_FRAC_W) - 1))) & 0xFFFF
    inexact = _fp16_bits_to_fraction(out) != value
    return Fp32ToFp16Result(bits, out, False, False, False, inexact, category)


def fraction_to_fp32(value):
    return fraction_to_fp32_bits(value)
