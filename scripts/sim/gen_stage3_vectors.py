import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.attention.single_head_reference import single_head_attention_bit_model  # noqa: E402
from model.attention.softmax_reference import fp32_exp, fp32_recip  # noqa: E402


PE_NUM = 8
FP16_ONE = 0x3C00
FP16_TWO = 0x4000
FP16_HALF = 0x3800
FP16_NEG_ONE = 0xBC00
FP16_NEG_HALF = 0xB800
FP16_ZERO = 0x0000


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def tile_outputs(output, d_head):
    tiles = []
    for base in range(0, d_head, PE_NUM):
        width = min(PE_NUM, d_head - base)
        mask = (1 << width) - 1
        values = output[base:base + width] + [0] * (PE_NUM - width)
        tiles.append((base, mask, values, int(base + PE_NUM >= d_head)))
    return tiles


def case_zero(seq_len, d_head):
    q = [FP16_ZERO for _ in range(d_head)]
    k = [[FP16_ZERO for _ in range(d_head)] for _ in range(seq_len)]
    v = [[FP16_ONE if tok == dim % seq_len else FP16_ZERO for dim in range(d_head)] for tok in range(seq_len)]
    return q, k, v


def case_uniform(seq_len, d_head):
    q = [FP16_ONE if idx % 2 == 0 else FP16_NEG_ONE for idx in range(d_head)]
    k_row = [FP16_HALF for _ in range(d_head)]
    k = [list(k_row) for _ in range(seq_len)]
    pattern = [FP16_ONE, FP16_TWO, FP16_NEG_ONE, FP16_HALF, FP16_NEG_HALF]
    v = [[pattern[(tok + dim) % len(pattern)] for dim in range(d_head)] for tok in range(seq_len)]
    return q, k, v


def case_one_hot(seq_len, d_head):
    q = [FP16_TWO for _ in range(d_head)]
    k = [[FP16_NEG_ONE for _ in range(d_head)] for _ in range(seq_len)]
    if seq_len > 1:
        k[seq_len // 2] = [FP16_TWO for _ in range(d_head)]
    v = [[FP16_HALF if tok % 2 else FP16_NEG_HALF for _ in range(d_head)] for tok in range(seq_len)]
    if seq_len > 1:
        v[seq_len // 2] = [FP16_TWO if dim % 2 == 0 else FP16_NEG_ONE for dim in range(d_head)]
    return q, k, v


def case_mixed(seq_len, d_head):
    q_pattern = [FP16_ONE, FP16_HALF, FP16_NEG_ONE, FP16_TWO, FP16_NEG_HALF, FP16_ZERO]
    k_pattern = [FP16_TWO, FP16_NEG_ONE, FP16_HALF, FP16_ONE, FP16_ZERO, FP16_NEG_HALF]
    v_pattern = [FP16_HALF, FP16_ONE, FP16_NEG_HALF, FP16_TWO, FP16_NEG_ONE, FP16_ZERO]
    q = [q_pattern[idx % len(q_pattern)] for idx in range(d_head)]
    k = [[k_pattern[(tok * 3 + dim) % len(k_pattern)] for dim in range(d_head)] for tok in range(seq_len)]
    v = [[v_pattern[(tok * 5 + dim * 2) % len(v_pattern)] for dim in range(d_head)] for tok in range(seq_len)]
    return q, k, v


def attention_lines(d_head):
    cases = [
        ("zero", 1, case_zero(1, d_head)),
        ("uniform", min(4, 32), case_uniform(min(4, 32), d_head)),
        ("onehot", min(7, 32), case_one_hot(min(7, 32), d_head)),
        ("mixed", min(8, 32), case_mixed(min(8, 32), d_head)),
    ]
    lines = []
    for case_idx, (name, seq_len, (q, k, v)) in enumerate(cases):
        trace = single_head_attention_bit_model(q, k, v, PE_NUM)
        meta = 0x400 + case_idx
        lines.append("CASE %s %d %04x" % (name, seq_len, meta))
        for dim, value in enumerate(q):
            lines.append("Q %d %04x" % (dim, value))
        for token, row in enumerate(k):
            for dim, value in enumerate(row):
                lines.append("K %d %d %04x" % (token, dim, value))
        for token, row in enumerate(v):
            for dim, value in enumerate(row):
                lines.append("V %d %d %04x" % (token, dim, value))
        for base, mask, values, last in tile_outputs(trace.output, d_head):
            fields = ["O", str(base), "%02x" % mask]
            fields.extend("%08x" % value for value in values)
            fields.append(str(last))
            lines.append(" ".join(fields))
        lines.append("RUN")
        lines.append("END")
    return lines


def exp_vectors():
    inputs = [
        0x00000000,
        0xBA83126F,
        0xBDCCCCCD,
        0xBF800000,
        0xC0A00000,
        0xC1200000,
        0xC1A00000,
        0xC1A80000,
    ]
    return ["%08x %08x %04x %01x" % (value, fp32_exp(value), 0x510 + idx, int(idx == len(inputs) - 1)) for idx, value in enumerate(inputs)]


def recip_vectors():
    inputs = [0x3F800000, 0x40000000, 0x40400000, 0x40800000, 0x41000000, 0x3E800000]
    return ["%08x %08x %04x %01x" % (value, fp32_recip(value), 0x610 + idx, int(idx == len(inputs) - 1)) for idx, value in enumerate(inputs)]


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: gen_stage3_vectors.py <output-dir>")
    out_dir = Path(sys.argv[1])
    write_lines(out_dir / "stage3_attention_d8.mem", attention_lines(8))
    write_lines(out_dir / "stage3_attention_d16.mem", attention_lines(16))
    write_lines(out_dir / "stage3_exp_vectors.mem", exp_vectors())
    write_lines(out_dir / "stage3_recip_vectors.mem", recip_vectors())
    print("stage3_attention_d8_cases=4")
    print("stage3_attention_d16_cases=4")
    print("stage3_exp_vectors=%d" % len(exp_vectors()))
    print("stage3_recip_vectors=%d" % len(recip_vectors()))


if __name__ == "__main__":
    main()

