#!/usr/bin/env python3
"""Generate Stage 7B RMSNorm and residual RTL vectors."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.arithmetic.fp16_fp32_reference import fp16_to_fp32_bits
from model.attention.softmax_reference import float_to_fp32_bits
from model.transformer.residual_reference import residual_add
from model.transformer.rmsnorm_reference import rmsnorm


FP16_ONE = 0x3C00
FP16_NEG_ONE = 0xBC00
FP16_HALF = 0x3800
FP16_TWO = 0x4000
FP16_NEG_HALF = 0xB800
FP16_QUARTER = 0x3400


def fp16_values_to_fp32(values):
    return [fp16_to_fp32_bits(value)["output_bits"] for value in values]


def make_case(d_model):
    if d_model == 8:
        input_fp16 = [0x0000, FP16_ONE, FP16_NEG_ONE, FP16_HALF, FP16_NEG_HALF, FP16_TWO, FP16_QUARTER, 0xB400]
        gamma_fp16 = [FP16_ONE, FP16_ONE, FP16_ONE, FP16_TWO, FP16_HALF, FP16_ONE, FP16_NEG_ONE, FP16_ONE]
        residual_lhs = [
            float_to_fp32_bits(1.0),
            float_to_fp32_bits(-1.0),
            float_to_fp32_bits(0.5),
            float_to_fp32_bits(-0.5),
            float_to_fp32_bits(2.0),
            float_to_fp32_bits(-2.0),
            float_to_fp32_bits(3.0),
            float_to_fp32_bits(-3.0),
        ]
        residual_rhs = [
            float_to_fp32_bits(-1.0),
            float_to_fp32_bits(1.0),
            float_to_fp32_bits(0.25),
            float_to_fp32_bits(-0.25),
            float_to_fp32_bits(-0.5),
            float_to_fp32_bits(0.5),
            float_to_fp32_bits(1.0),
            float_to_fp32_bits(-1.0),
        ]
    elif d_model == 16:
        input_fp16 = [
            FP16_ONE,
            FP16_ONE,
            FP16_ONE,
            FP16_ONE,
            FP16_NEG_ONE,
            FP16_NEG_ONE,
            FP16_NEG_ONE,
            FP16_NEG_ONE,
            FP16_HALF,
            FP16_NEG_HALF,
            FP16_TWO,
            FP16_NEG_HALF,
            0x0000,
            0x8000,
            FP16_QUARTER,
            0xB400,
        ]
        gamma_fp16 = [
            FP16_ONE,
            FP16_HALF,
            FP16_TWO,
            FP16_NEG_ONE,
            FP16_ONE,
            FP16_HALF,
            FP16_TWO,
            FP16_NEG_ONE,
            FP16_ONE,
            FP16_ONE,
            FP16_HALF,
            FP16_TWO,
            FP16_ONE,
            FP16_NEG_ONE,
            FP16_ONE,
            FP16_HALF,
        ]
        residual_lhs = [float_to_fp32_bits((idx - 7) * 0.25) for idx in range(d_model)]
        residual_rhs = [float_to_fp32_bits((7 - idx) * 0.125) for idx in range(d_model)]
    else:
        raise ValueError("unsupported d_model")

    input_fp32 = fp16_values_to_fp32(input_fp16)
    norm_trace = rmsnorm(input_fp32, gamma_fp16)
    residual_trace = residual_add(residual_lhs, residual_rhs)
    return input_fp32, gamma_fp16, norm_trace, residual_lhs, residual_rhs, residual_trace


def write_case(out_dir, name, d_model):
    path = out_dir / ("%s.mem" % name)
    input_fp32, gamma_fp16, norm_trace, residual_lhs, residual_rhs, residual_trace = make_case(d_model)
    lines = [
        "D %d" % d_model,
        "R %08x %08x" % (norm_trace.sum_sq, norm_trace.inv_rms),
    ]
    for dim in range(d_model):
        lines.append(
            "N %02x %08x %04x %04x"
            % (dim, input_fp32[dim], gamma_fp16[dim], norm_trace.norm_fp16[dim])
        )
    for dim in range(d_model):
        lines.append(
            "A %02x %08x %08x %08x"
            % (dim, residual_lhs[dim], residual_rhs[dim], residual_trace.output_fp32[dim])
        )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    print("%s lines=%d" % (path.name, len(lines)))


def main(argv):
    if len(argv) != 2:
        print("usage: gen_stage7b_vectors.py OUT_DIR", file=sys.stderr)
        return 2
    out_dir = Path(argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_case(out_dir, "stage7b_h1_d8", 8)
    write_case(out_dir, "stage7b_h2_d8", 16)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
