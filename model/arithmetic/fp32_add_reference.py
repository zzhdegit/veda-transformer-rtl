"""Bit-level FP32 add reference for Stage 2.

The model mirrors the Stage 2 FP32 add wrapper policy: finite FP32 operands are
added with round-to-nearest-even, NaN/Inf inputs are illegal, and exact zero
results use +0 to match the local DesignWare baseline style used by Stage 1B.
"""

from collections import namedtuple

from model.arithmetic.fp32_mac_reference import (
    fp32_bits_to_fraction,
    fraction_to_fp32_bits,
    is_supported_fp32_operand,
)


AddResult = namedtuple("AddResult", ["output_bits", "invalid"])


def fp32_add(a_bits, b_bits):
    a_bits &= 0xFFFFFFFF
    b_bits &= 0xFFFFFFFF
    invalid = not (is_supported_fp32_operand(a_bits) and is_supported_fp32_operand(b_bits))
    if invalid:
        return AddResult(0, True)

    value = fp32_bits_to_fraction(a_bits) + fp32_bits_to_fraction(b_bits)
    return AddResult(fraction_to_fp32_bits(value, zero_sign=0), False)
