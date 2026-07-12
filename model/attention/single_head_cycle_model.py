"""Simple Stage 3 single-head attention cycle estimate.

This model is intentionally conservative. It records the same counter names as
the RTL, but does not replace RTL measurement; report summaries use RTL counter
values when VCS is available.
"""

from collections import namedtuple


class AttentionCycleEstimate(namedtuple(
    "AttentionCycleEstimate",
    [
        "total_attention_cycles",
        "qk_cycles",
        "qk_pe_busy_cycles",
        "scale_cycles",
        "reduction_cycles",
        "reduction_finalize_cycles",
        "normalization_cycles",
        "sv_cycles",
        "pe_stall_cycles",
        "sfu_stall_cycles",
        "buffer_stall_cycles",
        "output_stall_cycles",
        "score_buffer_peak_occupancy",
    ],
)):

    def as_dict(self) -> dict:
        return self._asdict()


def estimate_attention_cycles(seq_len: int, d_head: int, pe_num: int) -> AttentionCycleEstimate:
    if seq_len <= 0 or d_head <= 0 or pe_num <= 0:
        raise ValueError("seq_len, d_head, and pe_num must be positive")
    qk_tiles = (d_head + pe_num - 1) // pe_num
    sv_tiles = qk_tiles
    # Stage 2 PE service time is dominated by transaction-serial conversion,
    # lane, reduction, and tile-add handshakes. This is a lower-confidence
    # static estimate; RTL counters are authoritative.
    pe_inner_tile_cost = 24 + 2 * (pe_num - 1)
    pe_outer_tile_cost = 8
    qk_cycles = seq_len * qk_tiles * pe_inner_tile_cost
    scale_cycles = seq_len * 2
    reduction_cycles = 1 + max(0, seq_len - 1) * 14
    normalization_cycles = 2 + seq_len * 8
    sv_cycles = sv_tiles * seq_len * pe_outer_tile_cost
    total = qk_cycles + scale_cycles + reduction_cycles + normalization_cycles + sv_cycles + sv_tiles
    return AttentionCycleEstimate(
        total_attention_cycles=total,
        qk_cycles=qk_cycles,
        qk_pe_busy_cycles=qk_cycles,
        scale_cycles=scale_cycles,
        reduction_cycles=reduction_cycles,
        reduction_finalize_cycles=1,
        normalization_cycles=normalization_cycles,
        sv_cycles=sv_cycles,
        pe_stall_cycles=0,
        sfu_stall_cycles=0,
        buffer_stall_cycles=0,
        output_stall_cycles=0,
        score_buffer_peak_occupancy=seq_len,
    )
