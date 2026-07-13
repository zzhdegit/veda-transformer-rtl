"""HW-H9 element-serial softmax reference schedule.

This module keeps the Stage 3 softmax arithmetic and models only the packet
movement needed by Hardware Stage H9: score production, score buffering,
online reduction, replay for normalization, probability FIFO, and ordered
probability delivery to sV.
"""

from collections import deque, namedtuple

from model.attention.softmax_reference import normalize_scores, online_softmax_reduction


ScorePacket = namedtuple(
    "ScorePacket",
    [
        "token_meta",
        "head_id",
        "logical_token_index",
        "cache_slot",
        "score_index",
        "tile_id",
        "lane_mask",
        "score_fp32",
        "last_in_tile",
        "last_in_head",
        "status",
        "invalid",
    ],
)


ProbabilityPacket = namedtuple(
    "ProbabilityPacket",
    [
        "token_meta",
        "head_id",
        "logical_token_index",
        "cache_slot",
        "probability_index",
        "probability_fp32",
        "last_probability",
        "status",
        "invalid",
    ],
)


SoftmaxStreamTrace = namedtuple(
    "SoftmaxStreamTrace",
    [
        "score_packets",
        "probability_packets",
        "max_score",
        "exp_sum",
        "inv_sum",
        "score_fifo_peak_occupancy",
        "probability_fifo_peak_occupancy",
        "score_fifo_full_stalls",
        "score_fifo_empty_cycles",
        "probability_fifo_full_stalls",
        "probability_fifo_empty_cycles",
    ],
)


def make_score_packets(scores, token_meta=0, head_id=0, cache_slots=None, tile_id=0, lane_mask=0):
    if cache_slots is None:
        cache_slots = list(range(len(scores)))
    if len(cache_slots) != len(scores):
        raise ValueError("cache_slots length must match scores")
    packets = []
    for index, score in enumerate(scores):
        packets.append(
            ScorePacket(
                token_meta=token_meta,
                head_id=head_id,
                logical_token_index=index,
                cache_slot=cache_slots[index],
                score_index=index,
                tile_id=tile_id,
                lane_mask=lane_mask,
                score_fp32=score & 0xFFFFFFFF,
                last_in_tile=True,
                last_in_head=index == len(scores) - 1,
                status=0,
                invalid=False,
            )
        )
    return packets


def run_interleaved_softmax(scores, score_fifo_depth=32, probability_fifo_depth=32, token_meta=0, head_id=0):
    """Return ordered probability packets using the frozen Stage 3 arithmetic.

    The arithmetic is intentionally the same as Stage 8. The stream model keeps
    bounded FIFO accounting instead of relying on an infinite Python queue.
    """
    if not scores:
        raise ValueError("softmax requires at least one score")
    if score_fifo_depth <= 0 or probability_fifo_depth <= 0:
        raise ValueError("FIFO depths must be positive")
    if len(scores) > score_fifo_depth or len(scores) > probability_fifo_depth:
        raise ValueError("FIFO depth must cover one head in this correctness model")

    score_fifo = deque()
    score_packets = []
    score_peak = 0
    score_full_stalls = 0
    score_empty_cycles = 0
    for packet in make_score_packets(scores, token_meta=token_meta, head_id=head_id):
        if len(score_fifo) >= score_fifo_depth:
            score_full_stalls += 1
        score_fifo.append(packet)
        score_packets.append(packet)
        score_peak = max(score_peak, len(score_fifo))

    reduction_scores = []
    while score_fifo:
        reduction_scores.append(score_fifo.popleft().score_fp32)
    if not reduction_scores:
        score_empty_cycles += 1

    reduction = online_softmax_reduction(reduction_scores)
    normalization = normalize_scores(reduction_scores, reduction["max"], reduction["exp_sum"])

    probability_fifo = deque()
    probability_packets = []
    probability_peak = 0
    probability_full_stalls = 0
    probability_empty_cycles = 0
    for index, probability in enumerate(normalization["probabilities"]):
        if len(probability_fifo) >= probability_fifo_depth:
            probability_full_stalls += 1
        packet = ProbabilityPacket(
            token_meta=token_meta,
            head_id=head_id,
            logical_token_index=index,
            cache_slot=index,
            probability_index=index,
            probability_fp32=probability & 0xFFFFFFFF,
            last_probability=index == len(scores) - 1,
            status=0,
            invalid=False,
        )
        probability_fifo.append(packet)
        probability_peak = max(probability_peak, len(probability_fifo))

    while probability_fifo:
        probability_packets.append(probability_fifo.popleft())
    if not probability_packets:
        probability_empty_cycles += 1

    return SoftmaxStreamTrace(
        score_packets=score_packets,
        probability_packets=probability_packets,
        max_score=reduction["max"],
        exp_sum=reduction["exp_sum"],
        inv_sum=normalization["inv_sum"],
        score_fifo_peak_occupancy=score_peak,
        probability_fifo_peak_occupancy=probability_peak,
        score_fifo_full_stalls=score_full_stalls,
        score_fifo_empty_cycles=score_empty_cycles,
        probability_fifo_full_stalls=probability_full_stalls,
        probability_fifo_empty_cycles=probability_empty_cycles,
    )
