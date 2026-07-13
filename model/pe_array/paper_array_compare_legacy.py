"""Compare Stage 8 paper-array bit model against the legacy PE model."""

import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.attention.softmax_reference import fp32_to_float
from model.pe.pe_core_reference import MODE_QK_INNER, inner_product_tiles, outer_product_sequence
from model.pe_array.paper_array_8x8x2_reference import PaperArray8x8x2Reference


def _ulp_distance(a_bits, b_bits):
    def ordered(bits):
        bits = bits & 0xFFFFFFFF
        if bits & 0x80000000:
            return (~bits + 1) & 0xFFFFFFFF
        return bits | 0x80000000

    return abs(ordered(a_bits) - ordered(b_bits))


def _vector_metrics(paper_bits, legacy_bits):
    paper = [fp32_to_float(value) for value in paper_bits]
    legacy = [fp32_to_float(value) for value in legacy_bits]
    diffs = [a - b for a, b in zip(paper, legacy)]
    abs_diffs = [abs(value) for value in diffs]
    if not paper:
        return {
            "max_abs_error": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "relative_l2": 0.0,
            "cosine_similarity": 1.0,
            "max_ulp": 0,
            "argmax_match": True,
        }
    norm_diff = math.sqrt(sum(value * value for value in diffs))
    norm_legacy = math.sqrt(sum(value * value for value in legacy))
    dot = sum(a * b for a, b in zip(paper, legacy))
    norm_paper = math.sqrt(sum(value * value for value in paper))
    denom = norm_paper * norm_legacy
    return {
        "max_abs_error": max(abs_diffs),
        "mae": sum(abs_diffs) / len(abs_diffs),
        "rmse": math.sqrt(sum(value * value for value in diffs) / len(diffs)),
        "relative_l2": norm_diff / norm_legacy if norm_legacy else 0.0,
        "cosine_similarity": dot / denom if denom else 1.0,
        "max_ulp": max(_ulp_distance(a, b) for a, b in zip(paper_bits, legacy_bits)),
        "argmax_match": paper.index(max(paper)) == legacy.index(max(legacy)),
    }


def compare_inner(q_fp16, k_fp16, legacy_pe_num=8):
    paper = PaperArray8x8x2Reference().inner_product(q_fp16, k_fp16)
    legacy, _ = inner_product_tiles(q_fp16, k_fp16, legacy_pe_num, MODE_QK_INNER)
    return {
        "paper": paper.scalar,
        "legacy": legacy,
        "bit_exact": paper.scalar == legacy,
        "metrics": _vector_metrics([paper.scalar], [legacy]),
    }


def compare_outer(probabilities_fp32, v_rows_fp16, legacy_pe_num=8):
    paper = PaperArray8x8x2Reference().outer_product(probabilities_fp32, v_rows_fp16)
    legacy = []
    vector_length = len(v_rows_fp16[0]) if v_rows_fp16 else 0
    for base in range(0, vector_length, legacy_pe_num):
        rows = [row[base:base + legacy_pe_num] for row in v_rows_fp16]
        masks = [(1 << min(legacy_pe_num, vector_length - base)) - 1 for _ in rows]
        acc, _ = outer_product_sequence(probabilities_fp32, rows, legacy_pe_num, masks)
        legacy.extend(acc[: min(legacy_pe_num, vector_length - base)])
    return {
        "paper": paper.vector,
        "legacy": legacy,
        "bit_exact": paper.vector == legacy,
        "metrics": _vector_metrics(paper.vector, legacy),
    }


def main():
    q = [0x3C00, 0x4000, 0xBC00, 0x3800, 0x3C00, 0xB800, 0x3400, 0x3000] * 2
    k = [0x4000, 0x3C00, 0x3800, 0xBC00, 0xB800, 0x3400, 0x3000, 0x3C00] * 2
    inner = compare_inner(q, k)
    print("Stage 8 paper-array vs legacy inner bit_exact=%s metrics=%s" % (inner["bit_exact"], inner["metrics"]))

    probabilities = [0x3F000000, 0x3F000000]
    rows = [q, k]
    outer = compare_outer(probabilities, rows)
    print("Stage 8 paper-array vs legacy outer bit_exact=%s metrics=%s" % (outer["bit_exact"], outer["metrics"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
