import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.cache.generation_reference import GenerationReference  # noqa: E402


PE_NUM = 8
MAX_SEQ_LEN = 8
FP16_ONE = 0x3C00
FP16_TWO = 0x4000
FP16_NEG_TWO = 0xC000
FP16_HALF = 0x3800
FP16_NEG_ONE = 0xBC00
FP16_NEG_HALF = 0xB800
FP16_ZERO = 0x0000


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def row(pattern, d_head, offset):
    return [pattern[(offset + idx) % len(pattern)] for idx in range(d_head)]


def token_stream(d_head, count):
    q_pattern = [FP16_TWO]
    k_winner = [FP16_TWO]
    k_loser = [FP16_NEG_TWO]
    v_pattern = [FP16_HALF, FP16_ONE, FP16_NEG_HALF, FP16_TWO, FP16_NEG_ONE, FP16_ZERO]
    tokens = []
    for step in range(count):
        tokens.append(
            {
                "q": row(q_pattern, d_head, step),
                "k": row(k_winner if step == 0 else k_loser, d_head, 0),
                "v": row(v_pattern, d_head, step * 5),
                "meta": 0x740 + step,
            }
        )
    return tokens


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


def generation_lines(d_head):
    ref = GenerationReference(d_head=d_head, max_seq_len=MAX_SEQ_LEN, pe_num=PE_NUM)
    tokens = token_stream(d_head, MAX_SEQ_LEN + 1)
    lines = []
    for step, token in enumerate(tokens):
        trace = ref.run_token(token["q"], token["k"], token["v"], token["meta"])
        lines.append(
            "STEP step%d %04x %d %d %d %02x"
            % (
                step,
                token["meta"],
                trace.seq_len_before,
                trace.seq_len_after,
                int(trace.invalid),
                trace.status,
            )
        )
        for dim in range(d_head):
            lines.append(
                "T %d %04x %04x %04x %d"
                % (
                    dim,
                    token["q"][dim],
                    token["k"][dim],
                    token["v"][dim],
                    int(dim == d_head - 1),
                )
            )
        for base, mask, values, last in tile_outputs(trace.output, d_head):
            fields = ["O", str(base), "%02x" % mask]
            fields.extend("%08x" % value for value in values)
            fields.append(str(last))
            lines.append(" ".join(fields))
        lines.append("RUN")
        lines.append("END")
    return lines


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: gen_stage4_vectors.py <output-dir>")
    out_dir = Path(sys.argv[1])
    write_lines(out_dir / "stage4_generation_d8.mem", generation_lines(8))
    write_lines(out_dir / "stage4_generation_d16.mem", generation_lines(16))
    print("stage4_generation_d8_steps=%d" % (MAX_SEQ_LEN + 1))
    print("stage4_generation_d16_steps=%d" % (MAX_SEQ_LEN + 1))


if __name__ == "__main__":
    main()
