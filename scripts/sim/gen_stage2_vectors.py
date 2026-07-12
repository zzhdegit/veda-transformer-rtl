import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.arithmetic.fp16_fp32_reference import fp16_to_fp32_bits  # noqa: E402
from model.arithmetic.fp32_add_reference import fp32_add  # noqa: E402
from model.arithmetic.fp32_mac_reference import fp32_mac  # noqa: E402
from model.pe.pe_core_reference import MODE_GEMV, MODE_QK_INNER, MODE_SV_OUTER, inner_product_tiles, outer_product_sequence  # noqa: E402
from model.pe.pe_lane_reference import PE_LANE_MODE_FMA, PE_LANE_MODE_PRODUCT, pe_lane_compute  # noqa: E402
from model.pe.reduction_tree_reference import balanced_reduction  # noqa: E402


PE_NUM = 8
FP16_ONE = 0x3C00
FP16_TWO = 0x4000
FP16_HALF = 0x3800
FP16_NEG_ONE = 0xBC00
FP16_NEG_HALF = 0xB800
FP16_ZERO = 0x0000


def f16(bits):
    return fp16_to_fp32_bits(bits)["output_bits"]


def pad16(values):
    return list(values) + [0] * (PE_NUM - len(values))


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def add_vectors():
    base = [
        (0x00000000, 0x00000000),
        (0x80000000, 0x00000000),
        (0x3F800000, 0x40000000),
        (0x3F800000, 0xBF800000),
        (0x7F7FFFFF, 0x00800000),
        (0x00800000, 0x80800000),
    ]
    lines = []
    state = 0x10203040
    for idx in range(24):
        state = ((state << 13) ^ state ^ (state >> 17) ^ (state << 5)) & 0xFFFFFFFF
        exp_a = 118 + (state % 18)
        frac_a = (state >> 7) & 0x007FFFFF
        a = ((state & 1) << 31) | (exp_a << 23) | frac_a
        state = ((state << 13) ^ state ^ (state >> 17) ^ (state << 5)) & 0xFFFFFFFF
        exp_b = 118 + (state % 18)
        frac_b = (state >> 5) & 0x007FFFFF
        b = (((state >> 1) & 1) << 31) | (exp_b << 23) | frac_b
        base.append((a, b))
    for idx, (a, b) in enumerate(base):
        result = fp32_add(a, b).output_bits
        lines.append("%08x %08x %08x %04x %01x" % (a, b, result, idx + 1, int(idx == len(base) - 1)))
    return lines


def lane_vectors():
    cases = [
        (PE_LANE_MODE_PRODUCT, 1, f16(FP16_TWO), f16(FP16_HALF), 0),
        (PE_LANE_MODE_PRODUCT, 1, f16(FP16_NEG_ONE), f16(FP16_TWO), 0),
        (PE_LANE_MODE_PRODUCT, 0, f16(FP16_NEG_ONE), f16(FP16_TWO), 0x3F800000),
        (PE_LANE_MODE_FMA, 1, 0x3F000000, f16(FP16_TWO), 0x3F800000),
        (PE_LANE_MODE_FMA, 1, 0xBF000000, f16(FP16_TWO), 0x40000000),
        (PE_LANE_MODE_FMA, 0, 0x7F7FFFFF, f16(FP16_ONE), 0x3F800000),
    ]
    lines = []
    for idx, (mode, active, scalar, vector, acc) in enumerate(cases):
        result = pe_lane_compute(mode, scalar, vector, acc, bool(active)).output_bits
        lines.append(
            "%01x %01x %08x %08x %08x %08x %04x %01x"
            % (mode, active, scalar, vector, acc, result, 0x100 + idx, int(idx == len(cases) - 1))
        )
    return lines


def reduction_vectors():
    raw = [
        [FP16_ONE, FP16_TWO, FP16_NEG_ONE, FP16_HALF, FP16_NEG_HALF, FP16_ZERO, FP16_ONE, FP16_ZERO],
        [FP16_ONE, FP16_NEG_ONE, FP16_ONE, FP16_NEG_ONE, FP16_ONE, FP16_NEG_ONE, FP16_ONE, FP16_NEG_ONE],
        [FP16_TWO, FP16_HALF, FP16_HALF, FP16_HALF, FP16_ZERO, FP16_ZERO, FP16_ZERO, FP16_ZERO],
        [FP16_ONE, FP16_ONE, FP16_ONE, FP16_ONE, FP16_ONE, FP16_ONE, FP16_ONE, FP16_ONE],
    ]
    masks = [0xFF, 0x07, 0x1F, 0x01]
    lines = []
    for idx, (values16, mask) in enumerate(zip(raw, masks)):
        values32 = [f16(v) for v in values16]
        result, invalid = balanced_reduction(values32, mask)
        assert not invalid
        fields = ["%02x" % mask] + ["%08x" % value for value in values32]
        fields += ["%08x" % result, "%04x" % (0x200 + idx), "%01x" % int(idx == len(raw) - 1)]
        lines.append(" ".join(fields))
    return lines


def core_vectors():
    lines = []

    def append_case(mode, clear, first, last, mask, scalar, vec_a, vec_b, expect, exp_scalar, exp_vector, meta, last_flag):
        fields = [
            "%01x" % mode,
            "%01x" % clear,
            "%01x" % first,
            "%01x" % last,
            "%02x" % mask,
            "%08x" % scalar,
        ]
        fields.extend("%04x" % value for value in pad16(vec_a))
        fields.extend("%04x" % value for value in pad16(vec_b))
        fields.append("%01x" % expect)
        fields.append("%08x" % exp_scalar)
        fields.extend("%08x" % value for value in list(exp_vector) + [0] * (PE_NUM - len(exp_vector)))
        fields.append("%04x" % meta)
        fields.append("%01x" % last_flag)
        lines.append(" ".join(fields))

    q1 = [FP16_ONE, FP16_TWO, FP16_NEG_ONE, FP16_HALF, FP16_ONE]
    k1 = [FP16_HALF, FP16_ONE, FP16_NEG_ONE, FP16_TWO, FP16_NEG_HALF]
    score1, _ = inner_product_tiles(q1, k1, PE_NUM, MODE_QK_INNER)
    append_case(MODE_QK_INNER, 1, 1, 1, 0x1F, 0, q1, k1, 1, score1, [], 0x301, 0)

    q2 = [FP16_ONE, FP16_HALF, FP16_NEG_ONE, FP16_TWO] * 4
    k2 = [FP16_TWO, FP16_NEG_ONE, FP16_HALF, FP16_ONE] * 4
    score2, _ = inner_product_tiles(q2[:13], k2[:13], PE_NUM, MODE_QK_INNER)
    append_case(MODE_QK_INNER, 1, 1, 0, 0xFF, 0, q2[:8], k2[:8], 0, 0, [], 0x302, 0)
    append_case(MODE_QK_INNER, 0, 0, 1, 0x1F, 0, q2[8:13], k2[8:13], 1, score2, [], 0x303, 0)

    probs = [0x3F800000, 0x3F000000, 0xBF000000]
    rows = [
        [FP16_ONE, FP16_TWO, FP16_NEG_ONE, FP16_HALF, FP16_ONE, FP16_ZERO, FP16_ONE, FP16_NEG_HALF],
        [FP16_TWO, FP16_ONE, FP16_ONE, FP16_NEG_ONE, FP16_HALF, FP16_ONE, FP16_ZERO, FP16_TWO],
        [FP16_ONE, FP16_ONE, FP16_TWO, FP16_ONE, FP16_ZERO, FP16_ONE, FP16_HALF, FP16_ONE],
    ]
    masks = [0xFF, 0x1F, 0x07]
    outer_acc, _ = outer_product_sequence(probs, rows, PE_NUM, masks)
    append_case(MODE_SV_OUTER, 1, 1, 0, masks[0], probs[0], [], rows[0], 0, 0, [], 0x304, 0)
    append_case(MODE_SV_OUTER, 0, 0, 0, masks[1], probs[1], [], rows[1], 0, 0, [], 0x305, 0)
    append_case(MODE_SV_OUTER, 0, 0, 1, masks[2], probs[2], [], rows[2], 1, 0, outer_acc, 0x306, 0)

    q3 = [FP16_ONE, FP16_TWO, FP16_HALF, FP16_NEG_HALF, FP16_ONE, FP16_ZERO, FP16_ONE, FP16_ONE]
    w3 = [FP16_ONE, FP16_HALF, FP16_TWO, FP16_ONE, FP16_NEG_ONE, FP16_ONE, FP16_ZERO, FP16_HALF]
    score3, _ = inner_product_tiles(q3, w3, PE_NUM, MODE_GEMV)
    append_case(MODE_GEMV, 1, 1, 1, 0xFF, 0, q3, w3, 1, score3, [], 0x307, 1)

    return lines


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: gen_stage2_vectors.py <output-dir>")
    out_dir = Path(sys.argv[1])
    write_lines(out_dir / "stage2_add_vectors.mem", add_vectors())
    write_lines(out_dir / "stage2_lane_vectors.mem", lane_vectors())
    write_lines(out_dir / "stage2_reduction_vectors.mem", reduction_vectors())
    write_lines(out_dir / "stage2_core_vectors.mem", core_vectors())
    print("stage2_add_vectors=%d" % len(add_vectors()))
    print("stage2_lane_vectors=%d" % len(lane_vectors()))
    print("stage2_reduction_vectors=%d" % len(reduction_vectors()))
    print("stage2_core_vectors=%d" % len(core_vectors()))


if __name__ == "__main__":
    main()
