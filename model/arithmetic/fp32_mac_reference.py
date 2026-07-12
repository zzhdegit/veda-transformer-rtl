"""Bit-level FP32 MAC reference for Stage 1B.

The model supports finite FP32 inputs, signed zero operands, subnormals,
round-to-nearest-even, and both fused and non-fused MAC semantics. It does not
use Python float arithmetic as the golden path. The Stage 1B DesignWare probe
showed that exact zero MAC results are returned as +0 for the selected RNE
rounding mode.
"""

from fractions import Fraction
from collections import namedtuple


FP32_EXP_W = 8
FP32_FRAC_W = 23
FP32_EXP_BIAS = 127
FP32_SIGNIFICAND_W = 24

ROUND_NEAREST_EVEN = 0

MacResult = namedtuple("MacResult", ["output_bits", "invalid", "semantics"])


def _mask(width):
    return (1 << width) - 1


def classify_fp32(bits):
    bits &= 0xFFFFFFFF
    exp = (bits >> FP32_FRAC_W) & _mask(FP32_EXP_W)
    frac = bits & _mask(FP32_FRAC_W)
    if exp == 0:
        return "zero" if frac == 0 else "subnormal"
    if exp == _mask(FP32_EXP_W):
        return "inf" if frac == 0 else "nan"
    return "normal"


def fp32_bits_to_fraction(bits):
    bits &= 0xFFFFFFFF
    sign = -1 if (bits >> 31) & 1 else 1
    exp = (bits >> FP32_FRAC_W) & _mask(FP32_EXP_W)
    frac = bits & _mask(FP32_FRAC_W)
    category = classify_fp32(bits)
    if category in ("inf", "nan"):
        raise ValueError("non-finite FP32 input")
    if exp == 0:
        if frac == 0:
            return Fraction(0, 1)
        return sign * Fraction(frac, 1 << 149)

    significand = (1 << FP32_FRAC_W) | frac
    exponent = exp - FP32_EXP_BIAS - FP32_FRAC_W
    if exponent >= 0:
        return sign * Fraction(significand << exponent, 1)
    return sign * Fraction(significand, 1 << (-exponent))


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
    numerator = value.numerator
    denominator = value.denominator
    quotient, remainder = divmod(numerator, denominator)
    twice = remainder * 2
    if twice > denominator:
        return quotient + 1
    if twice < denominator:
        return quotient
    return quotient + (quotient & 1)


def fraction_to_fp32_bits(value, zero_sign=0):
    if value == 0:
        return (zero_sign & 1) << 31

    sign = 1 if value < 0 else 0
    magnitude = -value if value < 0 else value
    min_normal = _pow2_fraction(-126)

    if magnitude < min_normal:
        sub = _round_fraction_to_int_rne(magnitude * (1 << 149))
        if sub == 0:
            return sign << 31
        if sub >= (1 << FP32_FRAC_W):
            return (sign << 31) | (1 << FP32_FRAC_W)
        return (sign << 31) | sub

    exponent = _floor_log2_positive(magnitude)
    scaled = magnitude / _pow2_fraction(exponent - FP32_FRAC_W)
    significand = _round_fraction_to_int_rne(scaled)
    if significand == (1 << FP32_SIGNIFICAND_W):
        significand >>= 1
        exponent += 1

    biased_exp = exponent + FP32_EXP_BIAS
    if biased_exp >= _mask(FP32_EXP_W):
        return (sign << 31) | (_mask(FP32_EXP_W) << FP32_FRAC_W)
    if biased_exp <= 0:
        sub = _round_fraction_to_int_rne(magnitude * (1 << 149))
        if sub == 0:
            return sign << 31
        return (sign << 31) | (sub & _mask(FP32_FRAC_W))

    frac = significand - (1 << FP32_FRAC_W)
    return ((sign << 31) | (biased_exp << FP32_FRAC_W) | (frac & _mask(FP32_FRAC_W))) & 0xFFFFFFFF


def is_supported_fp32_operand(bits):
    return classify_fp32(bits) not in ("inf", "nan")


def fp32_mac(a_bits, b_bits, c_bits, semantics="fused"):
    a_bits &= 0xFFFFFFFF
    b_bits &= 0xFFFFFFFF
    c_bits &= 0xFFFFFFFF
    invalid = not (
        is_supported_fp32_operand(a_bits)
        and is_supported_fp32_operand(b_bits)
        and is_supported_fp32_operand(c_bits)
    )
    if invalid:
        return MacResult(0, True, semantics)

    a = fp32_bits_to_fraction(a_bits)
    b = fp32_bits_to_fraction(b_bits)
    c = fp32_bits_to_fraction(c_bits)

    if semantics == "fused":
        result = fraction_to_fp32_bits(a * b + c, zero_sign=0)
    elif semantics == "non_fused":
        product_bits = fraction_to_fp32_bits(a * b)
        product = fp32_bits_to_fraction(product_bits)
        result = fraction_to_fp32_bits(product + c, zero_sign=0)
    else:
        raise ValueError("semantics must be fused or non_fused")

    return MacResult(result, False, semantics)


def find_fused_discriminator(limit=200000):
    # Deterministic xorshift search over normal finite values. The returned
    # tuple is (a, b, c, fused_result, non_fused_result).
    state = 0x13579BDF
    for _ in range(limit):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        a = 0x3F000000 | (state & 0x007FFFFF)
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        b = 0x3F000000 | (state & 0x007FFFFF)
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        c = 0xBF000000 | (state & 0x007FFFFF)
        fused = fp32_mac(a, b, c, "fused").output_bits
        non_fused = fp32_mac(a, b, c, "non_fused").output_bits
        if fused != non_fused:
            return a, b, c, fused, non_fused
    raise RuntimeError("no fused/non-fused discriminator found")
