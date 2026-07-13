"""Stage 7 FP32 residual reference."""

from collections import namedtuple

from model.arithmetic.fp32_add_reference import fp32_add


ResidualTrace = namedtuple("ResidualTrace", ["output_fp32", "invalid", "steps"])


def residual_add(lhs_fp32, rhs_fp32):
    if len(lhs_fp32) != len(rhs_fp32):
        raise ValueError("residual vector length mismatch")
    output = []
    steps = []
    invalid = False
    for dim, (lhs, rhs) in enumerate(zip(lhs_fp32, rhs_fp32)):
        add = fp32_add(lhs, rhs)
        output.append(add.output_bits)
        invalid = invalid or bool(add.invalid)
        steps.append({"dim": dim, "lhs": lhs & 0xFFFFFFFF, "rhs": rhs & 0xFFFFFFFF, "out": add.output_bits, "invalid": bool(add.invalid)})
    return ResidualTrace(output, invalid, steps)
