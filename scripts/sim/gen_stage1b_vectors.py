import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.arithmetic.fp32_mac_reference import (  # noqa: E402
    find_fused_discriminator,
    fp32_mac,
)


def _xorshift32(state):
    state ^= (state << 13) & 0xFFFFFFFF
    state ^= state >> 17
    state ^= (state << 5) & 0xFFFFFFFF
    return state & 0xFFFFFFFF


def _normal_from_state(state, negative=False):
    # Exponents 120..134 keep directed random vectors finite and simulation fast.
    exp = 120 + (state % 15)
    frac = (state >> 8) & 0x007FFFFF
    sign = 0x80000000 if negative else 0
    return sign | (exp << 23) | frac


def build_vectors():
    vectors = [
        (0x00000000, 0x3F800000, 0x00000000, 0x0001, 0),
        (0x80000000, 0x40000000, 0x3F800000, 0x0002, 0),
        (0x3FC00000, 0x40100000, 0x3F800000, 0x0003, 0),
        (0xBFC00000, 0x40100000, 0x3F800000, 0x0004, 0),
        (0x3F800001, 0x3F800000, 0xBF800000, 0x0005, 0),
        (0x7F7FFFFF, 0x3F000000, 0x7EFFFFFF, 0x0006, 0),
        (0x00800000, 0x3F000000, 0x00000000, 0x0007, 0),
        (0x00800000, 0x3F800000, 0x80800000, 0x0008, 0),
    ]

    disc = find_fused_discriminator()
    vectors.append((disc[0], disc[1], disc[2], 0x00F0, 0))

    state = 0x2468ACE1
    for idx in range(96):
        state = _xorshift32(state)
        a = _normal_from_state(state, negative=bool(state & 1))
        state = _xorshift32(state)
        b = _normal_from_state(state, negative=bool(state & 2))
        state = _xorshift32(state)
        c = _normal_from_state(state, negative=bool(state & 4))
        vectors.append((a, b, c, 0x0100 + idx, 1 if idx == 95 else 0))
    return vectors


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: gen_stage1b_vectors.py <output.mem>")

    out_path = Path(sys.argv[1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for a, b, c, meta, last in build_vectors():
        fused = fp32_mac(a, b, c, "fused").output_bits
        non_fused = fp32_mac(a, b, c, "non_fused").output_bits
        lines.append(
            "%08x %08x %08x %08x %08x %04x %01x"
            % (a, b, c, fused, non_fused, meta & 0xFFFF, last & 1)
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    print("stage1b_vectors=%d" % len(lines))


if __name__ == "__main__":
    main()
