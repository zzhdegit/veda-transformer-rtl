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
FP16_TWO = 0x4000
FP16_NEG_TWO = 0xC000
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
    values = [pool_value(offset + idx * 3) for idx in range(d_model)]
    values[0] = FP16_TWO
    values[1] = FP16_TWO if (offset % 2) == 0 else FP16_NEG_TWO
    return values


def identity(d_model):
    return [
        [FP16_ONE if out_idx == in_idx else FP16_ZERO for in_idx in range(d_model)]
        for out_idx in range(d_model)
    ]


def broadcast_column_weight(d_model, input_index):
    rows = []
    for _out_idx in range(d_model):
        rows.append([FP16_ONE if in_idx == input_index else FP16_ZERO for in_idx in range(d_model)])
    return rows


def tile_outputs(output, d_head):
    if not output:
        return []
    tiles = []
    for base in range(0, d_head, PE_NUM):
        width = min(PE_NUM, d_head - base)
        mask = (1 << width) - 1
        values = output[base:base + width] + [0] * (PE_NUM - width)
        tiles.append((base, mask, values, int(base + PE_NUM >= d_head)))
    return tiles


def vector_lines(n_head, d_head):
    d_model = n_head * d_head
    weights = {
        WQ: broadcast_column_weight(d_model, 0),
        WK: broadcast_column_weight(d_model, 1),
        WV: identity(d_model),
        WO: identity(d_model),
    }
    ref = ProjectionMhaReference(
        n_head=n_head,
        d_head=d_head,
        max_seq_len=MAX_SEQ_LEN,
        pe_num=PE_NUM,
        weights=weights,
    )
    lines = []
    for label, kind in [("WQ", WQ), ("WK", WK), ("WV", WV)]:
        lines.append(label)
        for row in weights[kind]:
            lines.append(" ".join("%04x" % value for value in row))

    for step in range(MAX_SEQ_LEN + 1):
        hidden = hidden_vector(d_model, step)
        meta = 0x6D00 + (n_head << 6) + step
        trace = ref.run_token(hidden, meta=meta)
        attention = trace.attention
        lines.append(
            "STEP step%d %04x %d %d %d %02x"
            % (
                step,
                meta,
                attention.seq_len_before,
                attention.seq_len_after,
                int(attention.invalid),
                attention.status,
            )
        )
        lines.append("H " + " ".join("%04x" % value for value in hidden))
        for head, output in enumerate(attention.outputs):
            for base, mask, values, last_tile in tile_outputs(output, d_head):
                fields = ["O", str(head), str(base), "%02x" % mask]
                fields.extend("%08x" % value for value in values)
                fields.extend([
                    str(last_tile),
                    str(int(last_tile and head == n_head - 1)),
                    str(int(last_tile and head == n_head - 1)),
                ])
                lines.append(" ".join(fields))
        lines.append("RUN")
        lines.append("END")
    return lines


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: gen_stage6d_vectors.py <output-dir>")
    out_dir = Path(sys.argv[1])
    configs = [
        ("h1_d8", 1, 8),
        ("h2_d8", 2, 8),
        ("h4_d8", 4, 8),
        ("h2_d16", 2, 16),
    ]
    for name, n_head, d_head in configs:
        lines = vector_lines(n_head, d_head)
        write_lines(out_dir / ("stage6d_projected_mha_%s.mem" % name), lines)
        print("stage6d_projected_mha_%s_lines=%d" % (name, len(lines)))


if __name__ == "__main__":
    main()
