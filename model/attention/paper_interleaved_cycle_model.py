"""HW-H9 paper Attention cycle and overlap model.

The default estimates are calibrated to the matched RTL A/B single-head
counter interval used by Hardware Stage H9. This is still a cycle-accounting
model, not a timing, frequency, area, or PPA result.
"""

from collections import namedtuple
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

H9CycleEstimate = namedtuple(
    "H9CycleEstimate",
    [
        "d_head",
        "seq_len",
        "staged_h8_cycles",
        "full_array_non_interleaved_cycles",
        "interleaved_cycles",
        "qk_cycles",
        "softmax_cycles",
        "sv_cycles",
        "qk_sfu_overlap_cycles",
        "sfu_sv_overlap_cycles",
        "qk_only_cycles",
        "sfu_only_cycles",
        "sv_only_cycles",
        "mode_switch_cycles",
        "pipeline_bubble_cycles",
        "score_fifo_peak_occupancy",
        "probability_fifo_peak_occupancy",
        "score_fifo_full_stall_cycles",
        "score_fifo_empty_cycles",
        "probability_fifo_full_stall_cycles",
        "probability_fifo_empty_stall_cycles",
        "array_active_cycles",
        "array_utilization",
        "sfu_active_cycles",
        "sfu_utilization",
    ],
)


def _tile_count(d_head):
    if d_head <= 0 or d_head > 128:
        raise ValueError("d_head must be in 1..128")
    return (d_head + 7) // 8


def _rtl_staged_cycles(d_head, seq_len):
    tiles = _tile_count(d_head)
    per_score_cycles = 69 * tiles + 15
    fixed_cycles = 5 * tiles + 14
    seq1_shortcut = 12 if seq_len == 1 else 0
    return per_score_cycles * seq_len + fixed_cycles - seq1_shortcut


def _rtl_interleaved_cycles(d_head, seq_len):
    tiles = _tile_count(d_head)
    return 65 * seq_len + 127 + 2 * tiles


def _native_full_array_non_interleaved_cycles(d_head, seq_len):
    tiles = _tile_count(d_head)
    qk_issue = 65 * seq_len
    sfu_replay = 36 * seq_len
    sv_issue = 23 * seq_len
    fixed = 78 + 2 * tiles
    return qk_issue + sfu_replay + sv_issue + fixed


def _qk_sfu_overlap(seq_len):
    return max(0, 14 * seq_len - 9)


def _sfu_sv_overlap(seq_len):
    return max(0, 5 * seq_len - 4)


def estimate_h9_interleaved_cycles(d_head, seq_len, score_fifo_depth=32, probability_fifo_depth=32):
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    tiles = _tile_count(d_head)
    staged = _rtl_staged_cycles(d_head, seq_len)
    interleaved = _rtl_interleaved_cycles(d_head, seq_len)
    full_array_non_interleaved = _native_full_array_non_interleaved_cycles(d_head, seq_len)
    qk_cycles = 32 * seq_len + 3 * tiles
    softmax_cycles = 27 * seq_len + 7
    sv_cycles = 6 * seq_len + tiles
    qk_sfu_overlap = _qk_sfu_overlap(seq_len)
    sfu_sv_overlap = _sfu_sv_overlap(seq_len)
    mode_switch_cycles = 2 + tiles
    array_active = qk_cycles + sv_cycles
    sfu_active = softmax_cycles
    bubble_cycles = max(
        0,
        interleaved
        - array_active
        - sfu_active
        + qk_sfu_overlap
        + sfu_sv_overlap,
    )
    score_peak = min(score_fifo_depth, max(1, min(seq_len, 4 + tiles)))
    prob_peak = min(probability_fifo_depth, max(1, min(seq_len, 2 + tiles // 2)))
    return H9CycleEstimate(
        d_head=d_head,
        seq_len=seq_len,
        staged_h8_cycles=staged,
        full_array_non_interleaved_cycles=full_array_non_interleaved,
        interleaved_cycles=interleaved,
        qk_cycles=qk_cycles,
        softmax_cycles=softmax_cycles,
        sv_cycles=sv_cycles,
        qk_sfu_overlap_cycles=qk_sfu_overlap,
        sfu_sv_overlap_cycles=sfu_sv_overlap,
        qk_only_cycles=max(0, qk_cycles - qk_sfu_overlap),
        sfu_only_cycles=max(0, softmax_cycles - qk_sfu_overlap - sfu_sv_overlap),
        sv_only_cycles=max(0, sv_cycles - sfu_sv_overlap),
        mode_switch_cycles=mode_switch_cycles,
        pipeline_bubble_cycles=bubble_cycles,
        score_fifo_peak_occupancy=score_peak,
        probability_fifo_peak_occupancy=prob_peak,
        score_fifo_full_stall_cycles=0,
        score_fifo_empty_cycles=max(0, 3 - min(seq_len, 3)),
        probability_fifo_full_stall_cycles=0,
        probability_fifo_empty_stall_cycles=max(0, 2 - min(seq_len, 2)),
        array_active_cycles=array_active,
        array_utilization=float(array_active) / float(interleaved),
        sfu_active_cycles=sfu_active,
        sfu_utilization=float(sfu_active) / float(interleaved),
    )


def main():
    for d_head in (8, 16, 64, 128):
        for seq_len in (1, 2, 8, 16, 32, 64):
            estimate = estimate_h9_interleaved_cycles(d_head, seq_len)
            print(
                "hw_h9_calibrated d_head=%d seq=%d staged=%d full_array_non_interleaved=%d interleaved=%d qk_sfu_overlap=%d sfu_sv_overlap=%d"
                % (
                    d_head,
                    seq_len,
                    estimate.staged_h8_cycles,
                    estimate.full_array_non_interleaved_cycles,
                    estimate.interleaved_cycles,
                    estimate.qk_sfu_overlap_cycles,
                    estimate.sfu_sv_overlap_cycles,
                )
            )


if __name__ == "__main__":
    main()
