import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.cache.multi_head_generation_reference import MultiHeadGenerationReference  # noqa: E402


PE_NUM = 8
DEFAULT_MAX_SEQ_LEN = 8
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


def token_stream(n_head, d_head, count):
    q_pattern = [FP16_TWO]
    k_winner = [FP16_TWO]
    k_loser = [FP16_NEG_TWO]
    v_pattern = [FP16_HALF, FP16_ONE, FP16_NEG_HALF, FP16_TWO, FP16_NEG_ONE, FP16_ZERO]
    tokens = []
    for step in range(count):
        q_heads = []
        k_heads = []
        v_heads = []
        for head in range(n_head):
            q_heads.append(row(q_pattern, d_head, step + head))
            k_heads.append(row(k_winner if step == 0 else k_loser, d_head, head))
            v_heads.append(row(v_pattern, d_head, step * 5 + head * 3))
        tokens.append(
            {
                "q": q_heads,
                "k": k_heads,
                "v": v_heads,
                "meta": 0x950 + (n_head << 6) + step,
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


def generation_lines(n_head, d_head, max_seq_len):
    ref = MultiHeadGenerationReference(n_head=n_head, d_head=d_head, max_seq_len=max_seq_len, pe_num=PE_NUM)
    tokens = token_stream(n_head, d_head, max_seq_len + 1)
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
        for head in range(n_head):
            for dim in range(d_head):
                last_dim = dim == d_head - 1
                last_head = head == n_head - 1 and last_dim
                lines.append(
                    "T %d %d %04x %04x %04x %d %d"
                    % (
                        head,
                        dim,
                        token["q"][head][dim],
                        token["k"][head][dim],
                        token["v"][head][dim],
                        int(last_dim),
                        int(last_head),
                    )
                )
        for head, output in enumerate(trace.outputs):
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


def parse_configs(text):
    configs = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        head_text, d_text = item.lower().split("x")
        configs.append((int(head_text), int(d_text)))
    return configs


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate Stage 5 multi-head generation RTL vectors.")
    parser.add_argument("output_dir")
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--configs", default="1x8,2x8,4x8,2x16")
    args = parser.parse_args(argv)
    if args.max_seq_len <= 0:
        raise SystemExit("--max-seq-len must be positive")
    out_dir = Path(args.output_dir)
    configs = parse_configs(args.configs)
    for n_head, d_head in configs:
        name = "stage5_generation_h%d_d%d.mem" % (n_head, d_head)
        write_lines(out_dir / name, generation_lines(n_head, d_head, args.max_seq_len))
        print("%s_steps=%d max_seq_len=%d" % (name, args.max_seq_len + 1, args.max_seq_len))


if __name__ == "__main__":
    main()
