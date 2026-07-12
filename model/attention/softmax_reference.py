"""Stage 3 FP32 softmax bit helpers.

The model follows the Stage 3 correctness baseline:

* all operands are finite FP32 values;
* subtraction is represented as FP32 add with the second operand's sign bit
  flipped;
* EXP clamps inputs below -20.0 to +0.0;
* reciprocal is modeled as round-to-FP32 1/x.

The EXP model intentionally uses the same finite, clamped semantics documented
for the RTL wrapper. It is not a formal proof of the vendor DW transcendental
implementation.
"""

import math
import struct

from model.arithmetic.fp32_add_reference import fp32_add
from model.arithmetic.fp32_mac_reference import (
    fp32_bits_to_fraction,
    fp32_mac,
    is_supported_fp32_operand,
)


FP32_ONE = 0x3F800000
FP32_ZERO = 0x00000000
FP32_EXP_MIN = 0xC1A00000  # -20.0


def fp32_to_float(bits: int) -> float:
    return struct.unpack(">f", struct.pack(">I", bits & 0xFFFFFFFF))[0]


def float_to_fp32_bits(value: float) -> int:
    return struct.unpack(">I", struct.pack(">f", float(value)))[0]


def fp32_neg(bits: int) -> int:
    return ((bits ^ 0x80000000) & 0xFFFFFFFF)


def fp32_sub(a_bits: int, b_bits: int) -> int:
    return fp32_add(a_bits, fp32_neg(b_bits)).output_bits


def fp32_gt(a_bits: int, b_bits: int) -> bool:
    a_bits &= 0xFFFFFFFF
    b_bits &= 0xFFFFFFFF
    if (a_bits & 0x7FFFFFFF) == 0 and (b_bits & 0x7FFFFFFF) == 0:
        return False
    a_sign = (a_bits >> 31) & 1
    b_sign = (b_bits >> 31) & 1
    if a_sign != b_sign:
        return b_sign == 1
    if a_sign == 0:
        return (a_bits & 0x7FFFFFFF) > (b_bits & 0x7FFFFFFF)
    return (a_bits & 0x7FFFFFFF) < (b_bits & 0x7FFFFFFF)


def fp32_max(a_bits: int, b_bits: int) -> int:
    return a_bits if fp32_gt(a_bits, b_bits) else b_bits


def fp32_exp(bits: int) -> int:
    bits &= 0xFFFFFFFF
    if not is_supported_fp32_operand(bits):
        return FP32_ZERO
    if fp32_gt(FP32_EXP_MIN, bits):
        return FP32_ZERO
    return float_to_fp32_bits(math.exp(fp32_to_float(bits)))


def fp32_recip(bits: int) -> int:
    bits &= 0xFFFFFFFF
    if not is_supported_fp32_operand(bits):
        return FP32_ZERO
    value = fp32_bits_to_fraction(bits)
    if value == 0:
        return FP32_ZERO
    return float_to_fp32_bits(float(1 / value))


def fp32_mul(a_bits: int, b_bits: int) -> int:
    return fp32_mac(a_bits, b_bits, FP32_ZERO, "fused").output_bits


def online_softmax_reduction(scaled_scores):
    if not scaled_scores:
        raise ValueError("softmax reduction requires at least one score")

    max_score = scaled_scores[0] & 0xFFFFFFFF
    exp_sum = FP32_ONE
    trace = [{"score": max_score, "max": max_score, "exp_sum": exp_sum}]

    for score in scaled_scores[1:]:
        score &= 0xFFFFFFFF
        new_max = fp32_max(max_score, score)
        old_delta = fp32_sub(max_score, new_max)
        x_delta = fp32_sub(score, new_max)
        old_exp = fp32_exp(old_delta)
        x_exp = fp32_exp(x_delta)
        scaled_old_sum = fp32_mul(exp_sum, old_exp)
        exp_sum = fp32_add(scaled_old_sum, x_exp).output_bits
        max_score = new_max
        trace.append(
            {
                "score": score,
                "max": max_score,
                "old_delta": old_delta,
                "x_delta": x_delta,
                "old_exp": old_exp,
                "x_exp": x_exp,
                "exp_sum": exp_sum,
            }
        )

    return {"max": max_score, "exp_sum": exp_sum, "trace": trace}


def normalize_scores(scaled_scores, max_score, exp_sum):
    inv_sum = fp32_recip(exp_sum)
    probabilities = []
    trace = []
    for score in scaled_scores:
        delta = fp32_sub(score, max_score)
        numerator = fp32_exp(delta)
        probability = fp32_mul(numerator, inv_sum)
        probabilities.append(probability)
        trace.append(
            {
                "score": score & 0xFFFFFFFF,
                "delta": delta,
                "numerator": numerator,
                "probability": probability,
            }
        )
    return {"inv_sum": inv_sum, "probabilities": probabilities, "trace": trace}


def probability_sum_float(probabilities):
    return sum(fp32_to_float(value) for value in probabilities)
