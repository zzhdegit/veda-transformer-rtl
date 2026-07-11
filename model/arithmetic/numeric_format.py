"""Bit-level helpers for Stage 1 signed integer and fixed-point references."""

from __future__ import annotations


def mask(width: int) -> int:
    if width <= 0:
        raise ValueError("width must be positive")
    return (1 << width) - 1


def bits_from_signed(value: int, width: int) -> int:
    return value & mask(width)


def signed_from_bits(bits: int, width: int) -> int:
    bits &= mask(width)
    sign = 1 << (width - 1)
    return bits - (1 << width) if bits & sign else bits


def sign_extend(value: int, from_width: int, to_width: int) -> int:
    if to_width < from_width:
        raise ValueError("to_width must be >= from_width")
    return signed_from_bits(bits_from_signed(value, from_width), from_width)


def fits_signed(value: int, width: int) -> bool:
    return signed_min(width) <= value <= signed_max(width)


def signed_min(width: int) -> int:
    if width <= 0:
        raise ValueError("width must be positive")
    return -(1 << (width - 1))


def signed_max(width: int) -> int:
    if width <= 0:
        raise ValueError("width must be positive")
    return (1 << (width - 1)) - 1


def saturate_signed(value: int, width: int) -> tuple[int, bool, bool]:
    if value > signed_max(width):
        return signed_max(width), True, False
    if value < signed_min(width):
        return signed_min(width), False, True
    return value, False, False


def truncate_signed(value: int, width: int) -> tuple[int, bool]:
    wrapped = signed_from_bits(bits_from_signed(value, width), width)
    return wrapped, wrapped != value


def round_shift_signed(value: int, frac_drop: int, mode: str) -> tuple[int, bool]:
    if frac_drop < 0:
        raise ValueError("frac_drop must be non-negative")
    if mode not in {"truncate", "nearest_even"}:
        raise ValueError(f"unsupported rounding mode: {mode}")
    if frac_drop == 0:
        return value, False

    scale = 1 << frac_drop
    remainder = abs(value) & (scale - 1)
    inexact = remainder != 0

    if mode == "truncate":
        # Match RTL arithmetic right shift semantics for two's-complement values.
        return value >> frac_drop, inexact

    magnitude = abs(value)
    truncated = magnitude >> frac_drop
    half = 1 << (frac_drop - 1)
    increment = remainder > half or (remainder == half and (truncated & 1) == 1)
    rounded_mag = truncated + int(increment)
    rounded = -rounded_mag if value < 0 else rounded_mag
    return rounded, inexact
