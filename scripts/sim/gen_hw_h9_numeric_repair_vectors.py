#!/usr/bin/env python3
"""Generate compact HW-H9-N1 numeric repair vectors."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.arithmetic.fp32_add_reference import fp32_add
from model.pe.pe_core_reference import MODE_GEMV, inner_product_tiles
from model.pe.reduction_tree_reference import balanced_reduction


PE_NUM = 8


def write(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def xorshift32(state):
    state ^= (state << 13) & 0xFFFFFFFF
    state ^= state >> 17
    state ^= (state << 5) & 0xFFFFFFFF
    return state & 0xFFFFFFFF


def random_fp32(state):
    state = xorshift32(state)
    sign = (state >> 31) & 1
    exp = 96 + (state % 48)
    frac = (state >> 7) & 0x007FFFFF
    return state, ((sign << 31) | (exp << 23) | frac) & 0xFFFFFFFF


def active_mask(width):
    return (1 << width) - 1 if width else 0


def add_vectors():
    pairs = [
        (0x3C81AA0C, 0x39699F40),
        (0x00000000, 0x00000000),
        (0x80000000, 0x00000000),
        (0x3F800000, 0xBF800000),
        (0x00800000, 0x80800000),
        (0x00000001, 0x00000001),
    ]
    state = 0x91EAD00D
    for _ in range(24):
        state, a = random_fp32(state)
        state, b = random_fp32(state)
        pairs.append((a, b))
    lines = []
    for idx, (a, b) in enumerate(pairs):
        result = fp32_add(a, b).output_bits
        lines.append("%08x %08x %08x %04x %01x" % (a, b, result, 0x9100 + idx, int(idx == len(pairs) - 1)))
    return lines


def reduction_vectors():
    directed = [
        (0xFF, [0xBBB498D0, 0, 0, 0xBA1B1720, 0x3C2CD660, 0x3C0DA134, 0x3C81AA0C, 0x39699F40]),
        (0x03, [0x3C81AA0C, 0x39699F40, 0, 0, 0, 0, 0, 0]),
        (0x0C, [0, 0, 0x3C81AA0C, 0x39699F40, 0, 0, 0, 0]),
        (0x30, [0, 0, 0, 0, 0x3C81AA0C, 0x39699F40, 0, 0]),
        (0xC0, [0, 0, 0, 0, 0, 0, 0x3C81AA0C, 0x39699F40]),
        (0xFF, [0x80000000, 0x00000000, 0x00000001, 0x00000001, 0x00800000, 0x80800000, 0x3F800000, 0xBF800000]),
    ]
    lines = []
    for idx, (mask, values) in enumerate(directed):
        result, invalid = balanced_reduction(values, mask)
        if invalid:
            raise RuntimeError("unexpected invalid directed reduction")
        fields = ["%02x" % mask] + ["%08x" % value for value in values]
        fields += ["%08x" % result, "%04x" % (0x9200 + idx), "%01x" % 0]
        lines.append(" ".join(fields))

    state = 0x31415926
    for idx in range(100):
        width = 1 + (idx % PE_NUM)
        mask = active_mask(width)
        values = []
        for lane in range(PE_NUM):
            state, value = random_fp32(state)
            values.append(value)
        result, invalid = balanced_reduction(values, mask)
        if invalid:
            raise RuntimeError("unexpected invalid random reduction")
        fields = ["%02x" % mask] + ["%08x" % value for value in values]
        fields += ["%08x" % result, "%04x" % (0x9300 + idx), "%01x" % int(idx == 99)]
        lines.append(" ".join(fields))
    return lines


def pad16(values):
    return list(values) + [0] * (PE_NUM - len(values))


def append_core_case(lines, mode, clear, first, last_tile, mask, a_values, b_values, expect, exp_scalar, meta, last_flag):
    fields = [
        "%01x" % mode,
        "%01x" % clear,
        "%01x" % first,
        "%01x" % last_tile,
        "%02x" % mask,
        "%08x" % 0,
    ]
    fields.extend("%04x" % value for value in pad16(a_values))
    fields.extend("%04x" % value for value in pad16(b_values))
    fields.append("%01x" % expect)
    fields.append("%08x" % exp_scalar)
    fields.extend("%08x" % 0 for _ in range(PE_NUM))
    fields.append("%04x" % meta)
    fields.append("%01x" % last_flag)
    lines.append(" ".join(fields))


def append_gemv_sequence(lines, a_values, b_values, meta, last_flag):
    result, tiles = inner_product_tiles(a_values, b_values, PE_NUM, MODE_GEMV)
    tile_count = len(tiles)
    for tile_idx in range(tile_count):
        base = tile_idx * PE_NUM
        width = min(PE_NUM, len(a_values) - base)
        expect = tile_idx == tile_count - 1
        append_core_case(
            lines,
            MODE_GEMV,
            int(tile_idx == 0),
            int(tile_idx == 0),
            int(expect),
            active_mask(width),
            a_values[base:base + width],
            b_values[base:base + width],
            int(expect),
            result if expect else 0,
            meta,
            last_flag if expect else 0,
        )


def core_vectors():
    lines = []
    # Artifact W2 row 1, base 8 tile. This is the first sensitive pair-3 path.
    act_base8 = [0x2D81, 0x0000, 0x0000, 0x344C, 0x3481, 0x2F0B, 0x367D, 0x3177]
    w2_base8 = [0xAC1A, 0x287F, 0x1EFC, 0x9883, 0x28CC, 0x2D07, 0x28FF, 0x1558]
    append_gemv_sequence(lines, act_base8, w2_base8, 0x9401, 0)

    # Artifact W2 consecutive base 0 and base 8 tiles; expected accumulator after base 8.
    act_base0 = [0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x3335, 0x3883, 0x2F46]
    w2_base0 = [0x187F, 0x23EF, 0x272C, 0xA688, 0x2975, 0xA832, 0xA11E, 0xAAC4]
    append_gemv_sequence(lines, act_base0 + act_base8, w2_base0 + w2_base8, 0x9402, 0)

    # Width coverage.
    append_gemv_sequence(lines, [0x3C00], [0x3C00], 0x9403, 0)
    append_gemv_sequence(lines, [0x3C00, 0xBC00], [0x3C00, 0x3C00], 0x9404, 0)
    append_gemv_sequence(lines, [0x3C00, 0x3800, 0xB800, 0x4000], [0x3800, 0x4000, 0x3C00, 0xBC00], 0x9405, 0)

    # D_MODEL=64 W1-like and D_FFN=256 W2-like coverage.
    state = 0xC0DEC0DE
    fp16_pool = [0x0000, 0x8000, 0x2C00, 0xAC00, 0x3000, 0xB000, 0x3400, 0xB400, 0x3800, 0xB800, 0x3C00, 0xBC00]
    a64 = []
    b64 = []
    for idx in range(64):
        state = xorshift32(state)
        a64.append(fp16_pool[state % len(fp16_pool)])
        state = xorshift32(state)
        b64.append(fp16_pool[state % len(fp16_pool)])
    append_gemv_sequence(lines, a64, b64, 0x9464, 0)
    a256 = []
    b256 = []
    for idx in range(256):
        state = xorshift32(state)
        a256.append(fp16_pool[state % len(fp16_pool)])
        state = xorshift32(state)
        b256.append(fp16_pool[state % len(fp16_pool)])
    append_gemv_sequence(lines, a256, b256, 0x9256, 1)
    return lines


def main(argv):
    if len(argv) != 2:
        print("usage: gen_hw_h9_numeric_repair_vectors.py OUT_DIR", file=sys.stderr)
        return 2
    out_dir = Path(argv[1])
    write(out_dir / "hw_h9_numeric_add.mem", add_vectors())
    write(out_dir / "hw_h9_numeric_reduction.mem", reduction_vectors())
    write(out_dir / "hw_h9_numeric_core.mem", core_vectors())
    print("hw_h9_numeric_add=%d" % len(add_vectors()))
    print("hw_h9_numeric_reduction=%d" % len(reduction_vectors()))
    print("hw_h9_numeric_core=%d" % len(core_vectors()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
