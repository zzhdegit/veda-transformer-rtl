"""Reference model for implemented Stage 1 integer arithmetic wrappers.

This is not an IEEE-754 model. It covers only the signed integer/fixed-point
helper behavior implemented in Stage 1 RTL while FP16/FP32 arithmetic IP
selection remains open.
"""

from __future__ import annotations

from dataclasses import dataclass

from .numeric_format import (
    bits_from_signed,
    fits_signed,
    round_shift_signed,
    saturate_signed,
    signed_from_bits,
    truncate_signed,
)


@dataclass(frozen=True)
class ArithmeticResult:
    input_bits: tuple[int, ...]
    output_bits: int
    high_precision_result: int
    rounded_result: int
    overflow: bool = False
    underflow: bool = False
    inexact: bool = False
    metadata: int | None = None


def signed_add(a_bits: int, b_bits: int, a_w: int, b_w: int, out_w: int) -> ArithmeticResult:
    a = signed_from_bits(a_bits, a_w)
    b = signed_from_bits(b_bits, b_w)
    total = a + b
    wrapped, overflow = truncate_signed(total, out_w)
    return ArithmeticResult(
        input_bits=(a_bits & ((1 << a_w) - 1), b_bits & ((1 << b_w) - 1)),
        output_bits=bits_from_signed(wrapped, out_w),
        high_precision_result=total,
        rounded_result=wrapped,
        overflow=overflow,
        underflow=overflow and total < 0,
    )


def signed_mul(a_bits: int, b_bits: int, a_w: int, b_w: int, out_w: int) -> ArithmeticResult:
    a = signed_from_bits(a_bits, a_w)
    b = signed_from_bits(b_bits, b_w)
    product = a * b
    wrapped, overflow = truncate_signed(product, out_w)
    return ArithmeticResult(
        input_bits=(a_bits & ((1 << a_w) - 1), b_bits & ((1 << b_w) - 1)),
        output_bits=bits_from_signed(wrapped, out_w),
        high_precision_result=product,
        rounded_result=wrapped,
        overflow=overflow,
        underflow=overflow and product < 0,
    )


def signed_mac(
    a_bits: int,
    b_bits: int,
    acc_bits: int,
    a_w: int,
    b_w: int,
    acc_w: int,
    clear: bool,
) -> ArithmeticResult:
    a = signed_from_bits(a_bits, a_w)
    b = signed_from_bits(b_bits, b_w)
    acc = signed_from_bits(acc_bits, acc_w)
    product = a * b
    total = product if clear else acc + product
    wrapped, overflow = truncate_signed(total, acc_w)
    return ArithmeticResult(
        input_bits=(
            a_bits & ((1 << a_w) - 1),
            b_bits & ((1 << b_w) - 1),
            acc_bits & ((1 << acc_w) - 1),
            int(clear),
        ),
        output_bits=bits_from_signed(wrapped, acc_w),
        high_precision_result=total,
        rounded_result=wrapped,
        overflow=overflow,
        underflow=overflow and total < 0,
    )


def compare_max(a_bits: int, b_bits: int, width: int) -> tuple[int, bool]:
    a = signed_from_bits(a_bits, width)
    b = signed_from_bits(b_bits, width)
    take_b = b > a
    value = b if take_b else a
    return bits_from_signed(value, width), take_b


def round_and_saturate(
    value_bits: int,
    in_w: int,
    out_w: int,
    frac_drop: int,
    rounding: str,
    saturate: bool = True,
) -> ArithmeticResult:
    value = signed_from_bits(value_bits, in_w)
    rounded, inexact = round_shift_signed(value, frac_drop, rounding)
    if saturate:
        final, overflow, underflow = saturate_signed(rounded, out_w)
    else:
        final, overflow = truncate_signed(rounded, out_w)
        underflow = overflow and rounded < 0

    return ArithmeticResult(
        input_bits=(value_bits & ((1 << in_w) - 1),),
        output_bits=bits_from_signed(final, out_w),
        high_precision_result=value,
        rounded_result=rounded,
        overflow=overflow or (not fits_signed(rounded, out_w)),
        underflow=underflow,
        inexact=inexact,
    )
