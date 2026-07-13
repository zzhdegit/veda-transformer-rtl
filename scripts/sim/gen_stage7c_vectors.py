#!/usr/bin/env python3
"""Generate Stage 7C FFN/ReLU RTL vectors."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.transformer.ffn_reference import ffn_forward


FP16_ZERO = 0x0000
FP16_ONE = 0x3C00
FP16_NEG_ONE = 0xBC00
FP16_HALF = 0x3800
FP16_NEG_HALF = 0xB800
FP16_QUARTER = 0x3400
FP16_NEG_QUARTER = 0xB400
FP16_TWO = 0x4000
FP16_NEG_TWO = 0xC000


def input_vector(d_model):
    base = [
        FP16_ONE,
        FP16_NEG_ONE,
        FP16_HALF,
        FP16_NEG_HALF,
        FP16_QUARTER,
        FP16_NEG_QUARTER,
        FP16_TWO,
        FP16_NEG_TWO,
    ]
    return [base[idx % len(base)] for idx in range(d_model)]


def make_weights(d_model):
    d_ffn = 4 * d_model
    w1 = []
    for row in range(d_ffn):
        values = [FP16_ZERO for _ in range(d_model)]
        primary = row % d_model
        secondary = (row + 3) % d_model
        if row % 4 == 0:
            values[primary] = FP16_ONE
        elif row % 4 == 1:
            values[primary] = FP16_NEG_ONE
        elif row % 4 == 2:
            values[primary] = FP16_HALF
            values[secondary] = FP16_HALF
        else:
            values[primary] = FP16_NEG_HALF
            values[secondary] = FP16_QUARTER
        w1.append(values)

    w2 = []
    for row in range(d_model):
        values = [FP16_ZERO for _ in range(d_ffn)]
        for col in range(d_ffn):
            if col % d_model == row:
                values[col] = FP16_HALF if (col // d_model) % 2 == 0 else FP16_NEG_QUARTER
            elif col % d_model == ((row + 1) % d_model) and (col // d_model) == 1:
                values[col] = FP16_QUARTER
        w2.append(values)
    return w1, w2


def write_case(out_dir, name, d_model):
    d_ffn = 4 * d_model
    inputs = input_vector(d_model)
    w1, w2 = make_weights(d_model)
    trace = ffn_forward(inputs, w1, w2, pe_num=8)
    lines = ["D %d %d" % (d_model, d_ffn)]
    for dim, value in enumerate(inputs):
        lines.append("I %02x %04x" % (dim, value))
    for row in range(d_ffn):
        for col in range(d_model):
            lines.append("W 0 %03x %03x %04x" % (row, col, w1[row][col]))
    for row in range(d_model):
        for col in range(d_ffn):
            lines.append("W 1 %03x %03x %04x" % (row, col, w2[row][col]))
    for dim, value in enumerate(trace.ffn2_fp32):
        lines.append("O %02x %08x" % (dim, value))
    path = out_dir / ("%s.mem" % name)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    print("%s lines=%d" % (path.name, len(lines)))


def main(argv):
    if len(argv) != 2:
        print("usage: gen_stage7c_vectors.py OUT_DIR", file=sys.stderr)
        return 2
    out_dir = Path(argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_case(out_dir, "stage7c_d8", 8)
    write_case(out_dir, "stage7c_d16", 16)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
