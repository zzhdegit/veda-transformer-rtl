"""Compare HW-H9 paper-native interleaved Attention to the H8 mapping."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.attention.paper_interleaved_attention_reference import h9_paper_interleaved_attention_bit_model
from model.pe_array.paper_array_compare_legacy import _vector_metrics


def compare_h9_to_h8(q_fp16, k_fp16, v_fp16):
    h9 = h9_paper_interleaved_attention_bit_model(q_fp16, k_fp16, v_fp16)
    h8 = h9.h8_comparison["paper"]
    return {
        "h9": h9,
        "h8": h8,
        "raw_scores_bit_exact": h9.raw_scores == h8.raw_scores,
        "scaled_scores_bit_exact": h9.scaled_scores == h8.scaled_scores,
        "probabilities_bit_exact": h9.probabilities == h8.probabilities,
        "output_bit_exact": h9.output == h8.output,
        "metrics": {
            "raw_scores": _vector_metrics(h9.raw_scores, h8.raw_scores),
            "scaled_scores": _vector_metrics(h9.scaled_scores, h8.scaled_scores),
            "probabilities": _vector_metrics(h9.probabilities, h8.probabilities),
            "output": _vector_metrics(h9.output, h8.output),
        },
    }


def deterministic_values(length):
    pool = [0x3C00, 0xBC00, 0x4000, 0x3800, 0xB800, 0x3400, 0xB400, 0x3000]
    return [pool[(index * 7 + 2) % len(pool)] for index in range(length)]


def main():
    for d_head in (8, 16, 64, 128):
        q = deterministic_values(d_head)
        k = []
        v = []
        for token in range(8):
            row = deterministic_values(d_head)
            if token % 2:
                row = list(reversed(row))
            k.append(row)
            v.append(row[token % d_head :] + row[: token % d_head])
        comparison = compare_h9_to_h8(q, k, v)
        print(
            "d_head=%d output_bit_exact=%s max_abs_error=%s max_ulp=%s"
            % (
                d_head,
                comparison["output_bit_exact"],
                comparison["metrics"]["output"]["max_abs_error"],
                comparison["metrics"]["output"]["max_ulp"],
            )
        )


if __name__ == "__main__":
    main()
