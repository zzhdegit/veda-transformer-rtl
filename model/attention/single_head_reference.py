"""Stage 3 single-head generation attention bit model."""

from collections import namedtuple

from model.arithmetic.fp16_fp32_reference import fp16_to_fp32_bits
from model.arithmetic.fp32_mac_reference import fp32_mac
from model.pe.pe_core_reference import MODE_QK_INNER, MODE_SV_OUTER, inner_product_tiles, outer_product_sequence
from model.attention.softmax_reference import (
    fp32_mul,
    fp32_to_float,
    normalize_scores,
    online_softmax_reduction,
)


SCALE_BY_D_HEAD = {
    1: 0x3F800000,
    7: 0x3EC1848F,
    8: 0x3EB504F3,
    9: 0x3EAAAAAB,
    13: 0x3E8E00D5,
    16: 0x3E800000,
    128: 0x3DB504F3,
}


SingleHeadBitTrace = namedtuple(
    "SingleHeadBitTrace",
    [
        "raw_scores",
        "scaled_scores",
        "max_score",
        "exp_sum",
        "inv_sum",
        "probabilities",
        "output",
        "qk_tiles",
        "sv_steps",
    ],
)


def scale_score(raw_score: int, d_head: int) -> int:
    if d_head not in SCALE_BY_D_HEAD:
        raise ValueError("unsupported D_HEAD scale constant: %d" % d_head)
    return fp32_mul(raw_score, SCALE_BY_D_HEAD[d_head])


def single_head_attention_bit_model(q_fp16, k_fp16, v_fp16, pe_num):
    if not q_fp16:
        raise ValueError("D_HEAD must be positive")
    if not k_fp16:
        raise ValueError("seq_len must be positive")
    if len(k_fp16) != len(v_fp16):
        raise ValueError("K/V sequence length mismatch")

    d_head = len(q_fp16)
    for row in k_fp16 + v_fp16:
        if len(row) != d_head:
            raise ValueError("K/V row dimension mismatch")

    raw_scores = []
    scaled_scores = []
    qk_tiles = []
    for k_row in k_fp16:
        raw, tiles = inner_product_tiles(q_fp16, k_row, pe_num, MODE_QK_INNER)
        raw_scores.append(raw)
        scaled_scores.append(scale_score(raw, d_head))
        qk_tiles.append(tiles)

    reduction = online_softmax_reduction(scaled_scores)
    normalization = normalize_scores(scaled_scores, reduction["max"], reduction["exp_sum"])

    output = [0 for _ in range(d_head)]
    sv_steps = []
    probabilities = normalization["probabilities"]
    for base in range(0, d_head, pe_num):
        rows = [row[base:base + pe_num] for row in v_fp16]
        masks = [(1 << min(pe_num, d_head - base)) - 1 for _ in rows]
        acc, steps = outer_product_sequence(probabilities, rows, pe_num, masks)
        for lane, value in enumerate(acc[: min(pe_num, d_head - base)]):
            output[base + lane] = value
        sv_steps.append(steps)

    return SingleHeadBitTrace(
        raw_scores=raw_scores,
        scaled_scores=scaled_scores,
        max_score=reduction["max"],
        exp_sum=reduction["exp_sum"],
        inv_sum=normalization["inv_sum"],
        probabilities=probabilities,
        output=output,
        qk_tiles=qk_tiles,
        sv_steps=sv_steps,
    )


def single_head_high_precision(q_fp16, k_fp16, v_fp16):
    import math

    q = [fp32_to_float(fp16_to_fp32_bits(value)["output_bits"]) for value in q_fp16]
    k = [[fp32_to_float(fp16_to_fp32_bits(value)["output_bits"]) for value in row] for row in k_fp16]
    v = [[fp32_to_float(fp16_to_fp32_bits(value)["output_bits"]) for value in row] for row in v_fp16]
    scale = 1.0 / math.sqrt(len(q))
    raw = [sum(a * b for a, b in zip(q, row)) for row in k]
    scaled = [value * scale for value in raw]
    max_score = max(scaled)
    exp_values = [math.exp(value - max_score) for value in scaled]
    exp_sum = sum(exp_values)
    probabilities = [value / exp_sum for value in exp_values]
    output = [sum(prob * row[dim] for prob, row in zip(probabilities, v)) for dim in range(len(q))]
    return {
        "raw_scores": raw,
        "scaled_scores": scaled,
        "max_score": max_score,
        "exp_sum": exp_sum,
        "probabilities": probabilities,
        "output": output,
    }


def output_error(bit_trace: SingleHeadBitTrace, high_precision: dict) -> dict:
    out_float = [fp32_to_float(value) for value in bit_trace.output]
    expected = high_precision["output"]
    abs_errors = [abs(a - b) for a, b in zip(out_float, expected)]
    rel_errors = [abs_err / max(abs(b), 1e-30) for abs_err, b in zip(abs_errors, expected)]
    return {
        "max_abs_error": max(abs_errors) if abs_errors else 0.0,
        "max_rel_error": max(rel_errors) if rel_errors else 0.0,
    }
