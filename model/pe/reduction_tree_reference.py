"""Deterministic balanced FP32 reduction tree model."""

from model.arithmetic.fp32_add_reference import fp32_add


def _is_power_of_two(value):
    return value > 0 and (value & (value - 1)) == 0


def balanced_reduction(values, lane_mask=None, trace=False):
    if not _is_power_of_two(len(values)):
        raise ValueError("balanced_reduction requires a power-of-two lane count")
    if lane_mask is None:
        lane_mask = (1 << len(values)) - 1

    current = [value & 0xFFFFFFFF if (lane_mask >> idx) & 1 else 0 for idx, value in enumerate(values)]
    levels = [list(current)]
    invalid = False
    while len(current) > 1:
        next_level = []
        for idx in range(0, len(current), 2):
            result = fp32_add(current[idx], current[idx + 1])
            invalid = invalid or bool(result.invalid)
            next_level.append(result.output_bits)
        current = next_level
        levels.append(list(current))

    if trace:
        return current[0], invalid, levels
    return current[0], invalid
