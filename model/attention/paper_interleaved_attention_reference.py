"""HW-H9 paper-native interleaved Attention bit model.

The model changes mapping and scheduling only. FP16 operands, FP32 products,
FP32 reductions, Stage 3 score scaling, and Stage 3 softmax arithmetic stay
unchanged.
"""

from collections import Counter, namedtuple
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.arithmetic.fp32_add_reference import fp32_add
from model.attention.paper_attention_reference import compare_paper_attention_to_legacy
from model.attention.paper_interleaved_softmax_reference import run_interleaved_softmax
from model.attention.single_head_reference import scale_score
from model.pe_array.paper_array_mapping import (
    ARRAY_COLS,
    ARRAY_GROUPS,
    ARRAY_ROWS,
    GROUP_CELLS,
    PE_CELLS,
    h9_native_cell_for_dim,
    h9_native_cell_index_for_dim,
)
from model.pe_array.paper_pe_group_reference import _reduce_eight
from model.pe_array.paper_pe_reference import PaperPECell


H9AttentionTrace = namedtuple(
    "H9AttentionTrace",
    [
        "raw_scores",
        "scaled_scores",
        "max_score",
        "exp_sum",
        "inv_sum",
        "probabilities",
        "output",
        "score_packets",
        "probability_packets",
        "active_pe_distribution",
        "group_active_counts",
        "row_active_counts",
        "column_active_counts",
        "h8_comparison",
        "mapping",
    ],
)


def _zero_grid():
    return [[[0 for _ in range(ARRAY_COLS)] for _ in range(ARRAY_ROWS)] for _ in range(ARRAY_GROUPS)]


def _make_cells():
    return [
        [[PaperPECell(group, row, column) for column in range(ARRAY_COLS)] for row in range(ARRAY_ROWS)]
        for group in range(ARRAY_GROUPS)
    ]


def _native_product_grid(a_fp16, b_fp16):
    cells = _make_cells()
    products = _zero_grid()
    active_counter = Counter()
    status = 0
    invalid = False
    for dim, (a_val, b_val) in enumerate(zip(a_fp16, b_fp16)):
        if dim >= PE_CELLS:
            raise ValueError("HW-H9 reference supports D_HEAD up to 128")
        group, row, column = h9_native_cell_for_dim(dim)
        out = cells[group][row][column].product(a_val, b_val, True)
        products[group][row][column] = out.product
        status |= out.status
        invalid = invalid or out.invalid
        active_counter[(group, row, column)] += 1
    return products, active_counter, status, invalid


def _reduce_native_grid(products):
    group_sums = []
    status = 0
    invalid = False
    for group in range(ARRAY_GROUPS):
        row_sums = []
        for row in range(ARRAY_ROWS):
            row_sum, row_status, row_invalid, _ = _reduce_eight(products[group][row])
            status |= row_status
            invalid = invalid or row_invalid
            row_sums.append(row_sum)
        group_sum, group_status, group_invalid, _ = _reduce_eight(row_sums)
        status |= group_status
        invalid = invalid or group_invalid
        group_sums.append(group_sum)
    combined = fp32_add(group_sums[0], group_sums[1])
    status |= 0
    invalid = invalid or bool(combined.invalid)
    return combined.output_bits, status, invalid


def h9_native_inner_product(a_fp16, b_fp16):
    if len(a_fp16) != len(b_fp16):
        raise ValueError("inner product operands must have equal length")
    if len(a_fp16) > PE_CELLS:
        raise ValueError("HW-H9 reference supports D_HEAD up to 128")
    products, active_counter, status, invalid = _native_product_grid(a_fp16, b_fp16)
    scalar, reduce_status, reduce_invalid = _reduce_native_grid(products)
    return {
        "scalar": scalar,
        "status": status | reduce_status,
        "invalid": invalid or reduce_invalid,
        "active_counter": active_counter,
    }


def h9_native_outer_product(probabilities_fp32, rows_fp16, vector_length):
    if len(probabilities_fp32) != len(rows_fp16):
        raise ValueError("probability and V row counts differ")
    if vector_length > PE_CELLS:
        raise ValueError("HW-H9 reference supports D_HEAD up to 128")
    for row in rows_fp16:
        if len(row) < vector_length:
            raise ValueError("V row shorter than vector length")

    cells = _make_cells()
    active_counter = Counter()
    status = 0
    invalid = False
    for probability, row_values in zip(probabilities_fp32, rows_fp16):
        for dim in range(vector_length):
            group, row, column = h9_native_cell_for_dim(dim)
            out = cells[group][row][column].outer_accumulate(probability, row_values[dim], True)
            status |= out.status
            invalid = invalid or out.invalid
            active_counter[(group, row, column)] += 1

    output = [0 for _ in range(vector_length)]
    for dim in range(vector_length):
        group, row, column = h9_native_cell_for_dim(dim)
        output[dim] = cells[group][row][column].accumulator
    return {
        "vector": output,
        "status": status,
        "invalid": invalid,
        "active_counter": active_counter,
    }


def _summarize_activity(counters):
    active_pe_distribution = Counter()
    group_active_counts = Counter()
    row_active_counts = Counter()
    column_active_counts = Counter()
    for counter in counters:
        for (group, row, column), count in counter.items():
            active_pe_distribution[count] += 1
            group_active_counts[group] += count
            row_active_counts[row] += count
            column_active_counts[column] += count
    return active_pe_distribution, group_active_counts, row_active_counts, column_active_counts


def h9_paper_interleaved_attention_bit_model(q_fp16, k_fp16, v_fp16, score_fifo_depth=32, probability_fifo_depth=32):
    if not q_fp16:
        raise ValueError("D_HEAD must be positive")
    if len(q_fp16) > PE_CELLS:
        raise ValueError("HW-H9 supports D_HEAD up to 128 in this stage")
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
    activity_counters = []
    for k_row in k_fp16:
        inner = h9_native_inner_product(q_fp16, k_row)
        raw_scores.append(inner["scalar"])
        scaled_scores.append(scale_score(inner["scalar"], d_head))
        activity_counters.append(inner["active_counter"])

    softmax = run_interleaved_softmax(
        scaled_scores,
        score_fifo_depth=score_fifo_depth,
        probability_fifo_depth=probability_fifo_depth,
    )
    probabilities = [packet.probability_fp32 for packet in softmax.probability_packets]
    outer = h9_native_outer_product(probabilities, v_fp16, d_head)
    activity_counters.append(outer["active_counter"])
    active_dist, group_counts, row_counts, col_counts = _summarize_activity(activity_counters)

    comparison = compare_paper_attention_to_legacy(q_fp16, k_fp16, v_fp16)
    mapping = [
        {
            "dimension": dim,
            "cell_index": h9_native_cell_index_for_dim(dim),
            "group_row_column": h9_native_cell_for_dim(dim),
        }
        for dim in range(d_head)
    ]
    return H9AttentionTrace(
        raw_scores=raw_scores,
        scaled_scores=scaled_scores,
        max_score=softmax.max_score,
        exp_sum=softmax.exp_sum,
        inv_sum=softmax.inv_sum,
        probabilities=probabilities,
        output=outer["vector"],
        score_packets=softmax.score_packets,
        probability_packets=softmax.probability_packets,
        active_pe_distribution=dict(active_dist),
        group_active_counts=dict(group_counts),
        row_active_counts=dict(row_counts),
        column_active_counts=dict(col_counts),
        h8_comparison=comparison,
        mapping=mapping,
    )


def main():
    q = [0x3C00, 0x4000, 0xBC00, 0x3800, 0x3C00, 0xB800, 0x3400, 0x3000]
    k = [q, list(reversed(q)), [0x3C00] * len(q)]
    v = [list(reversed(q)), q, [0x3800] * len(q)]
    trace = h9_paper_interleaved_attention_bit_model(q, k, v)
    print("h9_output=%s" % [hex(value) for value in trace.output])
    print("h9_group_active_counts=%s" % trace.group_active_counts)
    print("h9_h8_output_bit_exact=%s" % (trace.output == trace.h8_comparison["paper"].output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
