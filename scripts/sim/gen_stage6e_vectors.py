import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.projection.projection_mha_reference import ProjectionMhaReference  # noqa: E402
from model.projection.projection_reference import WQ, WK, WV, WO  # noqa: E402


PE_NUM = 8
MAX_SEQ_LEN = 8
FP16_ZERO = 0x0000
FP16_ONE = 0x3C00
FP16_NEG_ONE = 0xBC00
FP16_TWO = 0x4000
FP16_NEG_TWO = 0xC000
FP16_HALF = 0x3800
FP16_NEG_HALF = 0xB800
FP16_QUARTER = 0x3400
FP16_NEG_QUARTER = 0xB400
FP16_EIGHTH = 0x3000
FP16_NEG_EIGHTH = 0xB000


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def pool_value(index):
    pool = [FP16_ZERO, FP16_ONE, FP16_HALF, FP16_NEG_ONE, FP16_NEG_HALF, FP16_TWO, FP16_NEG_TWO]
    return pool[index % len(pool)]


def hidden_vector(d_model, offset):
    values = [pool_value(offset + idx * 3) for idx in range(d_model)]
    values[0] = FP16_TWO
    if d_model > 1:
        values[1] = FP16_TWO if (offset % 2) == 0 else FP16_NEG_TWO
    return values


def identity(d_model):
    return [
        [FP16_ONE if out_idx == in_idx else FP16_ZERO for in_idx in range(d_model)]
        for out_idx in range(d_model)
    ]


def broadcast_column_weight(d_model, input_index):
    return [
        [FP16_ONE if in_idx == input_index else FP16_ZERO for in_idx in range(d_model)]
        for _out_idx in range(d_model)
    ]


def dense_weight(d_model, salt):
    pool = [
        FP16_ZERO,
        FP16_EIGHTH,
        FP16_NEG_EIGHTH,
        FP16_QUARTER,
        FP16_NEG_QUARTER,
        FP16_HALF,
        FP16_NEG_HALF,
    ]
    rows = []
    for out_idx in range(d_model):
        row = []
        for in_idx in range(d_model):
            row.append(pool[(out_idx * 7 + in_idx * 5 + salt) % len(pool)])
        rows.append(row)
    return rows


def weights_for_config(n_head, d_head):
    d_model = n_head * d_head
    if n_head == 2 and d_head == 8:
        return {
            WQ: dense_weight(d_model, 1),
            WK: dense_weight(d_model, 2),
            WV: dense_weight(d_model, 3),
            WO: dense_weight(d_model, 4),
        }
    return {
        WQ: broadcast_column_weight(d_model, 0),
        WK: broadcast_column_weight(d_model, 1 if d_model > 1 else 0),
        WV: identity(d_model),
        WO: dense_weight(d_model, 4),
    }


def tile_outputs(output, d_model):
    if not output:
        return []
    tiles = []
    for base in range(0, d_model, PE_NUM):
        width = min(PE_NUM, d_model - base)
        mask = (1 << width) - 1
        values = output[base:base + width] + [0] * (PE_NUM - width)
        tiles.append((base, mask, values, int(base + PE_NUM >= d_model)))
    return tiles


def vector_lines(n_head, d_head):
    d_model = n_head * d_head
    weights = weights_for_config(n_head, d_head)
    ref = ProjectionMhaReference(
        n_head=n_head,
        d_head=d_head,
        max_seq_len=MAX_SEQ_LEN,
        pe_num=PE_NUM,
        weights=weights,
    )
    lines = []
    for label, kind in [("WQ", WQ), ("WK", WK), ("WV", WV), ("WO", WO)]:
        lines.append(label)
        for row in weights[kind]:
            lines.append(" ".join("%04x" % value for value in row))

    for step in range(MAX_SEQ_LEN + 1):
        hidden = hidden_vector(d_model, step)
        meta = 0x6E00 + (n_head << 6) + step
        trace = ref.run_token(hidden, meta=meta)
        lines.append(
            "STEP step%d %04x %d %d %d %02x"
            % (
                step,
                meta,
                trace.attention.seq_len_before,
                trace.attention.seq_len_after,
                int(trace.invalid),
                trace.status,
            )
        )
        lines.append("H " + " ".join("%04x" % value for value in hidden))
        for base, mask, values, last in tile_outputs(trace.final_output_fp32, d_model):
            fields = ["O", str(base), "%02x" % mask]
            fields.extend("%08x" % value for value in values)
            fields.append(str(last))
            lines.append(" ".join(fields))
        lines.append("RUN")
        lines.append("END")
    return lines


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: gen_stage6e_vectors.py <output-dir>")
    out_dir = Path(sys.argv[1])
    configs = [
        ("h1_d8", 1, 8),
        ("h2_d8", 2, 8),
        ("h4_d8", 4, 8),
        ("h2_d16", 2, 16),
    ]
    for name, n_head, d_head in configs:
        lines = vector_lines(n_head, d_head)
        write_lines(out_dir / ("stage6e_integrated_mha_%s.mem" % name), lines)
        print("stage6e_integrated_mha_%s_lines=%d" % (name, len(lines)))


if __name__ == "__main__":
    main()
