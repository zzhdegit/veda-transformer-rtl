import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.arithmetic.fp16_fp32_reference import fp16_to_fp32_bits  # noqa: E402
from model.arithmetic.fp32_mac_reference import fraction_to_fp32_bits  # noqa: E402
from model.projection.fp32_fp16_reference import fp32_to_fp16_bits  # noqa: E402
from model.projection.gemv_reference import gemv_output_row  # noqa: E402


PE_NUM = 8
D_MODEL = 32

FP16_ZERO = 0x0000
FP16_NEG_ZERO = 0x8000
FP16_ONE = 0x3C00
FP16_TWO = 0x4000
FP16_HALF = 0x3800
FP16_NEG_ONE = 0xBC00
FP16_NEG_HALF = 0xB800


def fp32(value):
    return fraction_to_fp32_bits(value)


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def xorshift(state):
    state ^= (state << 13) & 0xFFFFFFFF
    state ^= state >> 17
    state ^= (state << 5) & 0xFFFFFFFF
    return state & 0xFFFFFFFF


def fp32_to_fp16_vectors():
    values = [
        0x00000000,
        0x80000000,
        0x3F800000,
        0xBF800000,
        fp32(Fraction(1, 1) + Fraction(1, 2048)),
        fp32(Fraction(1, 1) + Fraction(1, 2048) + Fraction(1, 1 << 23)),
        fp32(Fraction(1, 1) + Fraction(3, 2048)),
        fp32(Fraction(4095, 2048)),
        fp32(Fraction(1, 1 << 14)),
        fp32(Fraction(1, 1 << 15)),
        fp32(Fraction(65504, 1)),
        fp32(Fraction(70000, 1)),
        0x00000001,
        0x80000001,
        0x7F800000,
        0xFF800000,
        0x7FC12345,
    ]
    state = 0x6B5A1234
    for _ in range(128):
        state = xorshift(state)
        sign = (state >> 31) & 1
        exp = 90 + (state % 70)
        frac = (state >> 5) & 0x7FFFFF
        values.append((sign << 31) | (exp << 23) | frac)
    lines = []
    for idx, value in enumerate(values):
        result = fp32_to_fp16_bits(value)
        lines.append(
            "%08x %04x %01x %01x %01x %01x %04x %01x"
            % (
                value,
                result.output_bits,
                int(result.invalid),
                int(result.overflow),
                int(result.underflow_or_ftz),
                int(result.inexact),
                0x6000 + idx,
                int(idx == len(values) - 1),
            )
        )
    return lines


def pattern(length, offset=0):
    values = [FP16_ONE, FP16_HALF, FP16_NEG_ONE, FP16_TWO, FP16_NEG_HALF, FP16_ZERO, FP16_ONE]
    return [values[(idx + offset) % len(values)] for idx in range(length)]


def pad(values):
    return list(values) + [0] * (D_MODEL - len(values))


def gemv_cases():
    cases = []
    cases.append(("len1", [FP16_TWO], [FP16_HALF]))
    cases.append(("len7", pattern(7), pattern(7, 2)))
    cases.append(("len8", pattern(8), pattern(8, 3)))
    cases.append(("len9", pattern(9), pattern(9, 4)))
    cases.append(("len16_zero", [FP16_ZERO] * 16, pattern(16)))
    cases.append(("len16_cancel", [FP16_ONE, FP16_NEG_ONE] * 8, [FP16_ONE] * 16))
    cases.append(("len32_random", pattern(32, 1), pattern(32, 5)))
    cases.append(("len32_identity_row", [FP16_ONE] + [FP16_ZERO] * 31, [FP16_ONE] + [FP16_ZERO] * 31))

    state = 0x12345678
    fp16_pool = [FP16_ZERO, FP16_ONE, FP16_HALF, FP16_NEG_ONE, FP16_NEG_HALF, FP16_TWO]
    for idx in range(8):
        length = [1, 7, 8, 9, 16, 17, 31, 32][idx]
        a = []
        b = []
        for _ in range(length):
            state = xorshift(state)
            a.append(fp16_pool[state % len(fp16_pool)])
            state = xorshift(state)
            b.append(fp16_pool[state % len(fp16_pool)])
        cases.append(("random%d" % idx, a, b))
    return cases


def gemv_vectors():
    lines = []
    for idx, (name, input_values, weight_values) in enumerate(gemv_cases()):
        trace = gemv_output_row(input_values, weight_values, PE_NUM, output_index=idx % D_MODEL)
        fields = [
            "%02x" % len(input_values),
            "%02x" % (idx % D_MODEL),
            "%04x" % (0x6800 + idx),
            "%08x" % trace.output_bits,
            "%01x" % int(trace.invalid),
            "%01x" % int(idx == len(gemv_cases()) - 1),
        ]
        fields.extend("%04x" % value for value in pad(input_values))
        fields.extend("%04x" % value for value in pad(weight_values))
        lines.append(" ".join(fields))
    return lines


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: gen_stage6b_vectors.py <output-dir>")
    out_dir = Path(sys.argv[1])
    fp_vectors = fp32_to_fp16_vectors()
    gemv = gemv_vectors()
    write_lines(out_dir / "stage6b_fp32_to_fp16.mem", fp_vectors)
    write_lines(out_dir / "stage6b_gemv.mem", gemv)
    print("stage6b_fp32_to_fp16_vectors=%d" % len(fp_vectors))
    print("stage6b_gemv_vectors=%d" % len(gemv))


if __name__ == "__main__":
    main()
