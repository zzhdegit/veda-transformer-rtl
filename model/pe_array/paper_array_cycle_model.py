"""Cycle estimates for the Stage 8 paper-structured PE array.

These are no-stall structural estimates for reporting only. They are not
frequency, timing, power, or area conclusions.
"""

from collections import namedtuple
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.pe_array.paper_array_mapping import PE_CELLS


PaperArrayCycleEstimate = namedtuple(
    "PaperArrayCycleEstimate",
    [
        "mode",
        "length",
        "tiles",
        "load_cycles",
        "compute_cycles",
        "l1_reduce_cycles",
        "l2_reduce_cycles",
        "output_cycles",
        "total_cycles",
    ],
)


def estimate_inner_cycles(length):
    tiles = (length + PE_CELLS - 1) // PE_CELLS
    if tiles == 0:
        tiles = 1
    load_cycles = tiles
    compute_cycles = tiles
    l1_reduce_cycles = 3 * tiles
    l2_reduce_cycles = 4 * tiles
    output_cycles = 1
    total = load_cycles + compute_cycles + l1_reduce_cycles + l2_reduce_cycles + output_cycles
    return PaperArrayCycleEstimate("inner", length, tiles, load_cycles, compute_cycles, l1_reduce_cycles, l2_reduce_cycles, output_cycles, total)


def estimate_outer_cycles(sequence_length, vector_length):
    tiles = (vector_length + PE_CELLS - 1) // PE_CELLS
    if tiles == 0:
        tiles = 1
    load_cycles = sequence_length * tiles
    compute_cycles = sequence_length * tiles
    l1_reduce_cycles = 0
    l2_reduce_cycles = 0
    output_cycles = tiles
    total = load_cycles + compute_cycles + output_cycles
    return PaperArrayCycleEstimate("outer", vector_length, tiles, load_cycles, compute_cycles, l1_reduce_cycles, l2_reduce_cycles, output_cycles, total)


def main():
    for length in [1, 7, 8, 9, 15, 16, 31, 32, 64, 128]:
        estimate = estimate_inner_cycles(length)
        print("inner length=%d tiles=%d total=%d" % (length, estimate.tiles, estimate.total_cycles))
    for length in [8, 16, 32, 64, 128]:
        estimate = estimate_outer_cycles(8, length)
        print("outer seq=8 length=%d tiles=%d total=%d" % (length, estimate.tiles, estimate.total_cycles))


if __name__ == "__main__":
    main()
