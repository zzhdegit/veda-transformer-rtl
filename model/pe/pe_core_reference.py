"""Stage 2 reconfigurable PE core bit model.

This model intentionally follows the frozen Stage 2 transaction rules:

* inner/GEMV tiles are reduced by the same balanced tree order as RTL;
* tile sums are accumulated sequentially in tile arrival order;
* outer-product updates write each active lane accumulator before the next
  transaction is accepted;
* lane masks convert inactive product lanes to +0 and hold inactive outer lanes.
"""

from model.arithmetic.fp16_fp32_reference import fp16_to_fp32_bits
from model.arithmetic.fp32_add_reference import fp32_add
from model.pe.pe_lane_reference import PE_LANE_MODE_FMA, PE_LANE_MODE_PRODUCT, pe_lane_compute
from model.pe.reduction_tree_reference import balanced_reduction


MODE_GEMV = 0
MODE_QK_INNER = 1
MODE_SV_OUTER = 2


def active_mask_for_width(width, pe_num):
    if width < 0 or width > pe_num:
        raise ValueError("active width out of range")
    return (1 << width) - 1 if width else 0


def inner_product_tiles(q_fp16, k_fp16, pe_num, mode=MODE_QK_INNER):
    if mode not in (MODE_GEMV, MODE_QK_INNER):
        raise ValueError("inner_product_tiles only supports GEMV/QK inner modes")
    if len(q_fp16) != len(k_fp16):
        raise ValueError("q and k lengths differ")

    acc = 0
    tiles = []
    for base in range(0, len(q_fp16), pe_num):
        q_tile = q_fp16[base:base + pe_num]
        k_tile = k_fp16[base:base + pe_num]
        width = len(q_tile)
        mask = active_mask_for_width(width, pe_num)
        products = []
        invalid = False
        for lane in range(pe_num):
            if lane < width:
                q32 = fp16_to_fp32_bits(q_tile[lane])["output_bits"]
                k32 = fp16_to_fp32_bits(k_tile[lane])["output_bits"]
            else:
                q32 = 0
                k32 = 0
            lane_result = pe_lane_compute(PE_LANE_MODE_PRODUCT, q32, k32, 0, bool((mask >> lane) & 1))
            invalid = invalid or bool(lane_result.invalid)
            products.append(lane_result.output_bits)

        tile_sum, red_invalid = balanced_reduction(products, mask)
        add_result = fp32_add(acc, tile_sum)
        acc = add_result.output_bits
        invalid = invalid or red_invalid or bool(add_result.invalid)
        tiles.append({"base": base, "mask": mask, "tile_sum": tile_sum, "acc": acc, "invalid": invalid})
    return acc, tiles


def outer_product_sequence(prob_fp32, v_fp16_rows, pe_num, lane_masks=None):
    if len(prob_fp32) != len(v_fp16_rows):
        raise ValueError("probability and V row lengths differ")
    if lane_masks is None:
        lane_masks = [(1 << min(pe_num, len(row))) - 1 for row in v_fp16_rows]

    acc = [0 for _ in range(pe_num)]
    steps = []
    for row_idx, (scalar, row, mask) in enumerate(zip(prob_fp32, v_fp16_rows, lane_masks)):
        invalid = False
        next_acc = list(acc)
        for lane in range(pe_num):
            v32 = fp16_to_fp32_bits(row[lane])["output_bits"] if lane < len(row) else 0
            lane_result = pe_lane_compute(
                PE_LANE_MODE_FMA,
                scalar,
                v32,
                acc[lane],
                bool((mask >> lane) & 1),
            )
            invalid = invalid or bool(lane_result.invalid)
            if (mask >> lane) & 1:
                next_acc[lane] = lane_result.output_bits
        acc = next_acc
        steps.append({"row": row_idx, "mask": mask, "acc": list(acc), "invalid": invalid})
    return acc, steps
