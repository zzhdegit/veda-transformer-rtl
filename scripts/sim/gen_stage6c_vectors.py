import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.projection.projection_reference import WQ, WK, WV, qkv_projection  # noqa: E402


FP16_ZERO = 0x0000
FP16_ONE = 0x3C00
FP16_TWO = 0x4000
FP16_HALF = 0x3800
FP16_NEG_ONE = 0xBC00
FP16_NEG_HALF = 0xB800


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def pool_value(index):
    pool = [FP16_ZERO, FP16_ONE, FP16_HALF, FP16_NEG_ONE, FP16_NEG_HALF, FP16_TWO]
    return pool[index % len(pool)]


def hidden_vector(d_model, offset):
    return [pool_value(offset + idx * 3) for idx in range(d_model)]


def weight_matrix(d_model, kind_offset):
    rows = []
    for out_idx in range(d_model):
        row = []
        for in_idx in range(d_model):
            if (out_idx + kind_offset) % d_model == in_idx:
                row.append(FP16_ONE)
            elif ((out_idx * 3 + in_idx + kind_offset) % 11) == 0:
                row.append(FP16_HALF)
            elif ((out_idx + in_idx + kind_offset) % 13) == 0:
                row.append(FP16_NEG_HALF)
            else:
                row.append(FP16_ZERO)
        rows.append(row)
    return rows


def vector_lines(n_head, d_head):
    d_model = n_head * d_head
    hidden = hidden_vector(d_model, n_head + d_head)
    weights = {
        WQ: weight_matrix(d_model, 1),
        WK: weight_matrix(d_model, 3),
        WV: weight_matrix(d_model, 5),
    }
    trace = qkv_projection(hidden, weights, n_head, d_head, pe_num=8)
    lines = []
    lines.append("HIDDEN " + " ".join("%04x" % value for value in hidden))
    for label, kind in [("WQ", WQ), ("WK", WK), ("WV", WV)]:
        lines.append(label)
        for row in weights[kind]:
            lines.append(" ".join("%04x" % value for value in row))
    lines.append("EXPECTED")
    for idx, item in enumerate(trace.qkv_stream):
        lines.append(
            "%02x %02x %02x %04x %04x %04x %01x %01x"
            % (
                idx,
                item["head"],
                item["dim"],
                item["q_fp16"],
                item["k_fp16"],
                item["v_fp16"],
                int(item["last_dim"]),
                int(item["last_head"]),
            )
        )
    return lines


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: gen_stage6c_vectors.py <output-dir>")
    out_dir = Path(sys.argv[1])
    configs = [
        ("h1_d8", 1, 8),
        ("h2_d8", 2, 8),
        ("h4_d8", 4, 8),
        ("h2_d16", 2, 16),
    ]
    for name, n_head, d_head in configs:
        lines = vector_lines(n_head, d_head)
        write_lines(out_dir / ("stage6c_qkv_%s.mem" % name), lines)
        print("stage6c_qkv_%s_lines=%d" % (name, len(lines)))


if __name__ == "__main__":
    main()
