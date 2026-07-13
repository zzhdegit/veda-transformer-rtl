"""Stage 8 paper-array attention bit model.

This model keeps the frozen Stage 3 softmax and FP32 arithmetic contract, but
routes QK and sV through the explicit 8x8x2 paper-array model.
"""

from collections import namedtuple
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.attention.single_head_reference import (
    scale_score,
    single_head_attention_bit_model,
)
from model.attention.softmax_reference import normalize_scores, online_softmax_reduction
from model.pe_array.paper_array_8x8x2_reference import PaperArray8x8x2Reference
from model.pe_array.paper_array_compare_legacy import _vector_metrics


PaperAttentionTrace = namedtuple(
    "PaperAttentionTrace",
    [
        "raw_scores",
        "scaled_scores",
        "max_score",
        "exp_sum",
        "inv_sum",
        "probabilities",
        "output",
        "qk_traces",
        "sv_traces",
        "array_counters",
    ],
)


def paper_single_head_attention_bit_model(q_fp16, k_fp16, v_fp16):
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

    array = PaperArray8x8x2Reference()
    raw_scores = []
    scaled_scores = []
    qk_traces = []
    for k_row in k_fp16:
        result = array.inner_product(q_fp16, k_row)
        raw_scores.append(result.scalar)
        scaled_scores.append(scale_score(result.scalar, d_head))
        qk_traces.append(result)

    reduction = online_softmax_reduction(scaled_scores)
    normalization = normalize_scores(scaled_scores, reduction["max"], reduction["exp_sum"])
    probabilities = normalization["probabilities"]

    sv_result = array.outer_product(probabilities, v_fp16, vector_length=d_head)
    return PaperAttentionTrace(
        raw_scores=raw_scores,
        scaled_scores=scaled_scores,
        max_score=reduction["max"],
        exp_sum=reduction["exp_sum"],
        inv_sum=normalization["inv_sum"],
        probabilities=probabilities,
        output=sv_result.vector,
        qk_traces=qk_traces,
        sv_traces=[sv_result],
        array_counters=array.counters(),
    )


def compare_paper_attention_to_legacy(q_fp16, k_fp16, v_fp16, legacy_pe_num=8):
    paper = paper_single_head_attention_bit_model(q_fp16, k_fp16, v_fp16)
    legacy = single_head_attention_bit_model(q_fp16, k_fp16, v_fp16, legacy_pe_num)
    raw_metrics = _vector_metrics(paper.raw_scores, legacy.raw_scores)
    scaled_metrics = _vector_metrics(paper.scaled_scores, legacy.scaled_scores)
    prob_metrics = _vector_metrics(paper.probabilities, legacy.probabilities)
    output_metrics = _vector_metrics(paper.output, legacy.output)
    return {
        "paper": paper,
        "legacy": legacy,
        "raw_scores_bit_exact": paper.raw_scores == legacy.raw_scores,
        "scaled_scores_bit_exact": paper.scaled_scores == legacy.scaled_scores,
        "probabilities_bit_exact": paper.probabilities == legacy.probabilities,
        "output_bit_exact": paper.output == legacy.output,
        "metrics": {
            "raw_scores": raw_metrics,
            "scaled_scores": scaled_metrics,
            "probabilities": prob_metrics,
            "output": output_metrics,
        },
    }


def main():
    q = [0x3C00, 0x4000, 0xBC00, 0x3800, 0x3C00, 0xB800, 0x3400, 0x3000]
    k = [q, list(reversed(q)), [0x3C00] * len(q)]
    v = [list(reversed(q)), q, [0x3800] * len(q)]
    comparison = compare_paper_attention_to_legacy(q, k, v)
    print("paper_attention_output_bit_exact=%s" % comparison["output_bit_exact"])
    print("paper_attention_output_metrics=%s" % comparison["metrics"]["output"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
