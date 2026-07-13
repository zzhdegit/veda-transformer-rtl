#!/usr/bin/env python3
"""Generate Stage 7D full transformer_layer RTL vectors."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.transformer.transformer_layer_reference import TransformerLayerReference
from scripts.sim.gen_stage7c_vectors import input_vector, make_weights, FP16_ZERO, FP16_ONE


def identity_matrix(d_model):
    return [
        [FP16_ONE if row == col else FP16_ZERO for col in range(d_model)]
        for row in range(d_model)
    ]


def token_vectors(d_model, token_count):
    first = input_vector(d_model)
    if token_count == 1:
        return [first]
    second = list(reversed(first))
    return [first, second]


def make_case(n_head, d_head, token_count=1):
    d_model = n_head * d_head
    hidden_tokens = token_vectors(d_model, token_count)
    gamma1 = [FP16_ONE for _ in range(d_model)]
    gamma2 = [FP16_ONE for _ in range(d_model)]
    w1, w2 = make_weights(d_model)
    mha_weights = {
        0: identity_matrix(d_model),
        1: identity_matrix(d_model),
        2: identity_matrix(d_model),
        3: identity_matrix(d_model),
    }
    ref = TransformerLayerReference(
        n_head=n_head,
        d_head=d_head,
        max_seq_len=8,
        pe_num=8,
        mha_weights=mha_weights,
        gamma1_fp16=gamma1,
        gamma2_fp16=gamma2,
        w1=w1,
        w2=w2,
    )
    traces = [ref.run_token(hidden, meta=0x7D01 + idx) for idx, hidden in enumerate(hidden_tokens)]
    return hidden_tokens, gamma1, gamma2, w1, w2, mha_weights, traces


def write_case(out_dir, name, n_head, d_head, token_count=1):
    d_model = n_head * d_head
    d_ffn = 4 * d_model
    hidden_tokens, gamma1, gamma2, w1, w2, mha_weights, traces = make_case(n_head, d_head, token_count)
    lines = ["C %d %d %d %d" % (n_head, d_head, d_model, d_ffn)]
    for kind in range(4):
        for row in range(d_model):
            for col in range(d_model):
                lines.append("W %d %03x %03x %04x" % (kind, row, col, mha_weights[kind][row][col]))
    for dim, value in enumerate(gamma1):
        lines.append("W 4 %03x 000 %04x" % (dim, value))
    for dim, value in enumerate(gamma2):
        lines.append("W 5 %03x 000 %04x" % (dim, value))
    for row in range(d_ffn):
        for col in range(d_model):
            lines.append("W 6 %03x %03x %04x" % (row, col, w1[row][col]))
    for row in range(d_model):
        for col in range(d_ffn):
            lines.append("W 7 %03x %03x %04x" % (row, col, w2[row][col]))
    for token_idx, (hidden, trace) in enumerate(zip(hidden_tokens, traces)):
        lines.append("T %d %04x" % (token_idx, 0x7D01 + token_idx))
        for dim, value in enumerate(hidden):
            lines.append("H %03x %04x" % (dim, value))
        for dim, value in enumerate(trace.final_fp32):
            lines.append("O %03x %08x" % (dim, value))
    path = out_dir / ("%s.mem" % name)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    print("%s lines=%d" % (path.name, len(lines)))


def main(argv):
    if len(argv) != 2:
        print("usage: gen_stage7d_vectors.py OUT_DIR", file=sys.stderr)
        return 2
    out_dir = Path(argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_case(out_dir, "stage7d_h1_d8", 1, 8)
    write_case(out_dir, "stage7d_h2_d8", 2, 8)
    write_case(out_dir, "stage7d_h4_d8", 4, 8)
    write_case(out_dir, "stage7d_h2_d16", 2, 16)
    write_case(out_dir, "stage7d_h2_d8_two_token", 2, 8, token_count=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
