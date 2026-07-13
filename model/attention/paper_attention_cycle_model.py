"""Stage 8 paper attention structural cycle model.

The estimates are no-stall bookkeeping baselines only. They are not frequency,
timing closure, area, or PPA conclusions.
"""

from collections import namedtuple
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.pe_array.paper_array_cycle_model import estimate_inner_cycles, estimate_outer_cycles


PaperAttentionCycleEstimate = namedtuple(
    "PaperAttentionCycleEstimate",
    [
        "d_head",
        "seq_len",
        "qk_cycles",
        "softmax_cycles",
        "sv_cycles",
        "mode_switch_cycles",
        "total_cycles",
    ],
)


def estimate_paper_attention_cycles(d_head, seq_len):
    if d_head <= 0 or seq_len <= 0:
        raise ValueError("d_head and seq_len must be positive")
    qk_one = estimate_inner_cycles(d_head)
    sv = estimate_outer_cycles(seq_len, d_head)
    qk_cycles = seq_len * qk_one.total_cycles
    softmax_cycles = seq_len * 2 + 4
    mode_switch_cycles = 1
    total = qk_cycles + softmax_cycles + sv.total_cycles + mode_switch_cycles
    return PaperAttentionCycleEstimate(
        d_head=d_head,
        seq_len=seq_len,
        qk_cycles=qk_cycles,
        softmax_cycles=softmax_cycles,
        sv_cycles=sv.total_cycles,
        mode_switch_cycles=mode_switch_cycles,
        total_cycles=total,
    )


def main():
    for d_head in (8, 16, 128):
        for seq_len in (1, 8):
            estimate = estimate_paper_attention_cycles(d_head, seq_len)
            print(
                "paper_attention d_head=%d seq_len=%d qk=%d softmax=%d sv=%d total=%d"
                % (
                    d_head,
                    seq_len,
                    estimate.qk_cycles,
                    estimate.softmax_cycles,
                    estimate.sv_cycles,
                    estimate.total_cycles,
                )
            )


if __name__ == "__main__":
    main()
