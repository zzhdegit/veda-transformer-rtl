"""HW-H9 paper Attention cycle and overlap model.

This is a structural schedule model. It is not a timing, frequency, area, or
PPA result. Latencies are modeled as small deterministic service times so the
producer/consumer overlap and bounded FIFO effects are explicit.
"""

from collections import namedtuple
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.attention.paper_attention_cycle_model import estimate_paper_attention_cycles


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


def _native_inner_latency(d_head):
    if d_head <= 0 or d_head > 128:
        raise ValueError("d_head must be in 1..128")
    return 10


def _native_outer_latency(d_head):
    if d_head <= 0 or d_head > 128:
        raise ValueError("d_head must be in 1..128")
    return 3


def _simulate_qk_sfu(seq_len, qk_latency, sfu_latency, fifo_depth):
    cycle = 0
    qk_remaining = 0
    sfu_remaining = 0
    qk_issued = 0
    qk_done = 0
    sfu_done = 0
    fifo_occ = 0
    peak = 0
    full_stalls = 0
    empty_cycles = 0
    qk_active_cycles = 0
    sfu_active_cycles = 0
    overlap_cycles = 0
    qk_only = 0
    sfu_only = 0

    while sfu_done < seq_len:
        qk_active = qk_remaining > 0
        sfu_active = sfu_remaining > 0
        if qk_active:
            qk_active_cycles += 1
        if sfu_active:
            sfu_active_cycles += 1
        if qk_active and sfu_active:
            overlap_cycles += 1
        elif qk_active:
            qk_only += 1
        elif sfu_active:
            sfu_only += 1

        if qk_remaining == 0 and qk_issued < seq_len:
            qk_remaining = qk_latency
            qk_issued += 1

        if sfu_remaining == 0:
            if fifo_occ > 0:
                fifo_occ -= 1
                sfu_remaining = sfu_latency
            elif qk_done < seq_len:
                empty_cycles += 1

        if qk_remaining > 0:
            qk_remaining -= 1
            if qk_remaining == 0:
                qk_done += 1
                if fifo_occ >= fifo_depth:
                    full_stalls += 1
                else:
                    fifo_occ += 1
                    peak = max(peak, fifo_occ)

        if sfu_remaining > 0:
            sfu_remaining -= 1
            if sfu_remaining == 0:
                sfu_done += 1

        cycle += 1

    return {
        "cycles": cycle,
        "qk_active_cycles": qk_active_cycles,
        "sfu_active_cycles": sfu_active_cycles,
        "overlap_cycles": overlap_cycles,
        "qk_only": qk_only,
        "sfu_only": sfu_only,
        "peak": peak,
        "full_stalls": full_stalls,
        "empty_cycles": empty_cycles,
    }


def _simulate_sfu_sv(seq_len, norm_latency, sv_latency, fifo_depth):
    cycle = 0
    norm_remaining = 0
    sv_remaining = 0
    norm_issued = 0
    norm_done = 0
    sv_done = 0
    fifo_occ = 0
    peak = 0
    full_stalls = 0
    empty_cycles = 0
    norm_active_cycles = 0
    sv_active_cycles = 0
    overlap_cycles = 0
    sfu_only = 0
    sv_only = 0

    while sv_done < seq_len:
        norm_active = norm_remaining > 0
        sv_active = sv_remaining > 0
        if norm_active:
            norm_active_cycles += 1
        if sv_active:
            sv_active_cycles += 1
        if norm_active and sv_active:
            overlap_cycles += 1
        elif norm_active:
            sfu_only += 1
        elif sv_active:
            sv_only += 1

        if norm_remaining == 0 and norm_issued < seq_len:
            norm_remaining = norm_latency
            norm_issued += 1

        if sv_remaining == 0:
            if fifo_occ > 0:
                fifo_occ -= 1
                sv_remaining = sv_latency
            elif norm_done < seq_len:
                empty_cycles += 1

        if norm_remaining > 0:
            norm_remaining -= 1
            if norm_remaining == 0:
                norm_done += 1
                if fifo_occ >= fifo_depth:
                    full_stalls += 1
                else:
                    fifo_occ += 1
                    peak = max(peak, fifo_occ)

        if sv_remaining > 0:
            sv_remaining -= 1
            if sv_remaining == 0:
                sv_done += 1

        cycle += 1

    return {
        "cycles": cycle,
        "norm_active_cycles": norm_active_cycles,
        "sv_active_cycles": sv_active_cycles,
        "overlap_cycles": overlap_cycles,
        "sfu_only": sfu_only,
        "sv_only": sv_only,
        "peak": peak,
        "full_stalls": full_stalls,
        "empty_cycles": empty_cycles,
    }


def estimate_h9_interleaved_cycles(d_head, seq_len, score_fifo_depth=32, probability_fifo_depth=32):
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    qk_latency = _native_inner_latency(d_head)
    reduction_latency = 8
    norm_latency = 8
    sv_latency = _native_outer_latency(d_head)
    mode_switch_cycles = 1
    final_recip_cycles = 2

    h8 = estimate_paper_attention_cycles(d_head, seq_len)
    qk_sfu = _simulate_qk_sfu(seq_len, qk_latency, reduction_latency, score_fifo_depth)
    sfu_sv = _simulate_sfu_sv(seq_len, norm_latency, sv_latency, probability_fifo_depth)
    interleaved = qk_sfu["cycles"] + final_recip_cycles + mode_switch_cycles + sfu_sv["cycles"]
    full_array_non_interleaved = seq_len * qk_latency + seq_len * reduction_latency + final_recip_cycles + mode_switch_cycles + seq_len * norm_latency + seq_len * sv_latency
    array_active = seq_len * qk_latency + seq_len * sv_latency
    sfu_active = qk_sfu["sfu_active_cycles"] + sfu_sv["norm_active_cycles"] + final_recip_cycles
    return H9CycleEstimate(
        d_head=d_head,
        seq_len=seq_len,
        staged_h8_cycles=h8.total_cycles,
        full_array_non_interleaved_cycles=full_array_non_interleaved,
        interleaved_cycles=interleaved,
        qk_cycles=seq_len * qk_latency,
        softmax_cycles=qk_sfu["sfu_active_cycles"] + final_recip_cycles + sfu_sv["norm_active_cycles"],
        sv_cycles=seq_len * sv_latency,
        qk_sfu_overlap_cycles=qk_sfu["overlap_cycles"],
        sfu_sv_overlap_cycles=sfu_sv["overlap_cycles"],
        qk_only_cycles=qk_sfu["qk_only"],
        sfu_only_cycles=qk_sfu["sfu_only"] + sfu_sv["sfu_only"],
        sv_only_cycles=sfu_sv["sv_only"],
        mode_switch_cycles=mode_switch_cycles,
        pipeline_bubble_cycles=max(0, interleaved - array_active - sfu_active + qk_sfu["overlap_cycles"] + sfu_sv["overlap_cycles"]),
        score_fifo_peak_occupancy=qk_sfu["peak"],
        probability_fifo_peak_occupancy=sfu_sv["peak"],
        score_fifo_full_stall_cycles=qk_sfu["full_stalls"],
        score_fifo_empty_cycles=qk_sfu["empty_cycles"],
        probability_fifo_full_stall_cycles=sfu_sv["full_stalls"],
        probability_fifo_empty_stall_cycles=sfu_sv["empty_cycles"],
        array_active_cycles=array_active,
        array_utilization=float(array_active) / float(interleaved),
        sfu_active_cycles=sfu_active,
        sfu_utilization=float(sfu_active) / float(interleaved),
    )


def main():
    for d_head in (8, 16, 64, 128):
        for seq_len in (1, 2, 8, 16, 32):
            estimate = estimate_h9_interleaved_cycles(d_head, seq_len)
            print(
                "hw_h9 d_head=%d seq=%d h8=%d full_array=%d h9=%d qk_sfu_overlap=%d sfu_sv_overlap=%d"
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
