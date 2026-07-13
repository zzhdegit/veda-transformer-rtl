"""One 8x8 PE group for the Stage 8 paper-structured array."""

from collections import namedtuple

from model.arithmetic.fp32_add_reference import fp32_add
from model.pe_array.paper_array_mapping import ARRAY_COLS, ARRAY_ROWS, GROUP_CELLS, cell_linear_index
from model.pe_array.paper_pe_reference import PaperPECell, type_mapping_legal


GroupResult = namedtuple(
    "GroupResult",
    [
        "value",
        "vector",
        "status",
        "invalid",
        "active_cells",
        "l1_trace",
        "l2_trace",
        "cell_trace",
    ],
)


def _add_pair(a_bits, b_bits):
    result = fp32_add(a_bits, b_bits)
    return result.output_bits, 0, bool(result.invalid)


def _reduce_eight(values):
    """Repository-defined Figure 5(d) order for an 8-input row/tree."""
    trace = []
    status = 0
    invalid = False
    pair01, st, inv = _add_pair(values[0], values[1])
    status |= st
    invalid = invalid or inv
    trace.append(("pair_1_2", pair01))
    pair23, st, inv = _add_pair(values[2], values[3])
    status |= st
    invalid = invalid or inv
    trace.append(("pair_3_4", pair23))
    pair45, st, inv = _add_pair(values[4], values[5])
    status |= st
    invalid = invalid or inv
    trace.append(("pair_5_6", pair45))
    pair67, st, inv = _add_pair(values[6], values[7])
    status |= st
    invalid = invalid or inv
    trace.append(("pair_7_8", pair67))

    half0, st, inv = _add_pair(pair01, pair23)
    status |= st
    invalid = invalid or inv
    trace.append(("quad_1_4", half0))
    half1, st, inv = _add_pair(pair45, pair67)
    status |= st
    invalid = invalid or inv
    trace.append(("quad_5_8", half1))

    total, st, inv = _add_pair(half0, half1)
    status |= st
    invalid = invalid or inv
    trace.append(("oct_1_8", total))
    return total, status, invalid, trace


class PaperPEGroup(object):
    """A physical 8x8 PE group with explicit cells and reductions."""

    def __init__(self, group_index):
        self.group_index = group_index
        self.cells = []
        for row in range(ARRAY_ROWS):
            for column in range(ARRAY_COLS):
                self.cells.append(PaperPECell(group_index, row, column))
        if len(self.cells) != GROUP_CELLS:
            raise AssertionError("paper PE group must have 64 cells")
        if not type_mapping_legal(self.cells):
            raise AssertionError("paper PE group type mapping is illegal")

    def reset(self):
        for cell in self.cells:
            cell.reset()

    def cell(self, row, column):
        return self.cells[row * ARRAY_COLS + column]

    def inner_product_tile(self, operands_a_fp16, operands_b_fp16, tile_base, vector_length, active=True, row_mask=0xFF, column_mask=0xFF):
        products_by_row = []
        cell_trace = []
        l1_trace = []
        status = 0
        invalid = False
        active_cells = 0

        for row in range(ARRAY_ROWS):
            row_products = []
            for column in range(ARRAY_COLS):
                global_index = cell_linear_index(self.group_index, row, column)
                dim = tile_base + global_index
                cell_active = active and dim < vector_length and bool((row_mask >> row) & 1) and bool((column_mask >> column) & 1)
                operand_a = operands_a_fp16[dim] if cell_active else 0
                operand_b = operands_b_fp16[dim] if cell_active else 0
                out = self.cell(row, column).product(operand_a, operand_b, cell_active)
                cell_trace.append(out)
                row_products.append(out.product if cell_active else 0)
                if cell_active:
                    active_cells += 1
                    status |= out.status
                    invalid = invalid or out.invalid
            row_sum, row_status, row_invalid, row_trace = _reduce_eight(row_products)
            status |= row_status
            invalid = invalid or row_invalid
            products_by_row.append(row_sum)
            l1_trace.append({"row": row, "trace": row_trace, "sum": row_sum})

        group_sum, l2_status, l2_invalid, l2_trace = _reduce_eight(products_by_row)
        status |= l2_status
        invalid = invalid or l2_invalid
        return GroupResult(group_sum, [], status, invalid, active_cells, l1_trace, l2_trace, cell_trace)

    def clear_outer_accumulators(self):
        for cell in self.cells:
            cell.accumulator = 0
            cell.forwarded_partial = 0

    def outer_product_step(self, scalar_fp32, vector_b_fp16, tile_base, vector_length, active=True, row_mask=0xFF, column_mask=0xFF):
        vector = [0 for _ in range(GROUP_CELLS)]
        cell_trace = []
        status = 0
        invalid = False
        active_cells = 0
        for row in range(ARRAY_ROWS):
            for column in range(ARRAY_COLS):
                local_index = row * ARRAY_COLS + column
                global_index = cell_linear_index(self.group_index, row, column)
                dim = tile_base + global_index
                cell_active = active and dim < vector_length and bool((row_mask >> row) & 1) and bool((column_mask >> column) & 1)
                operand_b = vector_b_fp16[dim] if cell_active else 0
                out = self.cell(row, column).outer_accumulate(scalar_fp32, operand_b, cell_active)
                cell_trace.append(out)
                vector[local_index] = out.accumulator
                if cell_active:
                    active_cells += 1
                    status |= out.status
                    invalid = invalid or out.invalid
        return GroupResult(0, vector, status, invalid, active_cells, [], [], cell_trace)
