"""Bit-accurate Stage 8 8x8x2 paper-structured PE array model."""

from collections import namedtuple

from model.arithmetic.fp32_add_reference import fp32_add
from model.pe_array.paper_array_mapping import (
    ARRAY_GROUPS,
    GROUP_CELLS,
    MODE_INNER_PRODUCT,
    MODE_OUTER_PRODUCT,
    PE_CELLS,
    default_column_mask,
    default_group_mask,
    default_row_mask,
    tile_bases,
)
from model.pe_array.paper_pe_group_reference import PaperPEGroup


ArrayResult = namedtuple(
    "ArrayResult",
    [
        "mode",
        "scalar",
        "vector",
        "status",
        "invalid",
        "active_cells",
        "tile_traces",
        "mode_switch",
    ],
)


class PaperArray8x8x2Reference(object):
    """Explicit 128-cell Stage 8 array model."""

    def __init__(self):
        self.groups = [PaperPEGroup(group) for group in range(ARRAY_GROUPS)]
        self.last_mode = None
        self.mode_switch_count = 0
        self.command_count = 0
        self.reset_count = 0
        self.active_cycles = 0
        self.idle_cycles = 0

    @property
    def cells(self):
        all_cells = []
        for group in self.groups:
            all_cells.extend(group.cells)
        return all_cells

    def reset(self):
        for group in self.groups:
            group.reset()
        self.last_mode = None
        self.mode_switch_count = 0
        self.command_count = 0
        self.active_cycles = 0
        self.idle_cycles = 0
        self.reset_count += 1

    def _record_command(self, mode):
        mode_switch = self.last_mode is not None and self.last_mode != mode
        if mode_switch:
            self.mode_switch_count += 1
        self.last_mode = mode
        self.command_count += 1
        return mode_switch

    def cell_count(self):
        return len(self.cells)

    def inner_product(self, operands_a_fp16, operands_b_fp16, group_mask=None, row_mask=None, column_mask=None):
        if len(operands_a_fp16) != len(operands_b_fp16):
            raise ValueError("inner product operands must have equal length")
        if group_mask is None:
            group_mask = default_group_mask()
        if row_mask is None:
            row_mask = default_row_mask()
        if column_mask is None:
            column_mask = default_column_mask()

        mode_switch = self._record_command(MODE_INNER_PRODUCT)
        total = 0
        status = 0
        invalid = False
        active_cells = 0
        tile_traces = []
        vector_length = len(operands_a_fp16)

        for tile_base in tile_bases(vector_length, PE_CELLS):
            group_sums = []
            group_traces = []
            for group_index, group in enumerate(self.groups):
                group_active = bool((group_mask >> group_index) & 1)
                result = group.inner_product_tile(
                    operands_a_fp16,
                    operands_b_fp16,
                    tile_base,
                    vector_length,
                    group_active,
                    row_mask,
                    column_mask,
                )
                group_sums.append(result.value if group_active else 0)
                group_traces.append(result)
                status |= result.status
                invalid = invalid or result.invalid
                active_cells += result.active_cells

            combined, st, inv = self._combine_groups(group_sums)
            status |= st
            invalid = invalid or inv
            add_result = fp32_add(total, combined)
            total = add_result.output_bits
            invalid = invalid or bool(add_result.invalid)
            tile_traces.append({"tile_base": tile_base, "groups": group_traces, "combined": combined, "acc": total})

        self.active_cycles += max(1, len(tile_traces))
        return ArrayResult(MODE_INNER_PRODUCT, total, [], status, invalid, active_cells, tile_traces, mode_switch)

    def outer_product(self, scalars_fp32, rows_fp16, vector_length=None, group_mask=None, row_mask=None, column_mask=None):
        if len(scalars_fp32) != len(rows_fp16):
            raise ValueError("outer product scalar and row counts differ")
        if vector_length is None:
            vector_length = len(rows_fp16[0]) if rows_fp16 else 0
        for row in rows_fp16:
            if len(row) < vector_length:
                raise ValueError("outer product row shorter than vector length")
        if group_mask is None:
            group_mask = default_group_mask()
        if row_mask is None:
            row_mask = default_row_mask()
        if column_mask is None:
            column_mask = default_column_mask()

        mode_switch = self._record_command(MODE_OUTER_PRODUCT)
        output = [0 for _ in range(vector_length)]
        status = 0
        invalid = False
        active_cells = 0
        tile_traces = []

        for tile_base in tile_bases(vector_length, PE_CELLS):
            for group in self.groups:
                group.clear_outer_accumulators()
            step_traces = []
            for scalar, row_values in zip(scalars_fp32, rows_fp16):
                group_results = []
                for group_index, group in enumerate(self.groups):
                    group_active = bool((group_mask >> group_index) & 1)
                    result = group.outer_product_step(
                        scalar,
                        row_values,
                        tile_base,
                        vector_length,
                        group_active,
                        row_mask,
                        column_mask,
                    )
                    group_results.append(result)
                    status |= result.status
                    invalid = invalid or result.invalid
                    active_cells += result.active_cells
                step_traces.append(group_results)

            for group_index, group in enumerate(self.groups):
                if not bool((group_mask >> group_index) & 1):
                    continue
                for local_index, cell_value in enumerate([cell.accumulator for cell in group.cells]):
                    dim = tile_base + group_index * GROUP_CELLS + local_index
                    if dim < vector_length:
                        output[dim] = cell_value
            tile_traces.append({"tile_base": tile_base, "steps": step_traces, "output": list(output)})

        self.active_cycles += max(1, len(tile_traces) * max(1, len(scalars_fp32)))
        return ArrayResult(MODE_OUTER_PRODUCT, 0, output, status, invalid, active_cells, tile_traces, mode_switch)

    def _combine_groups(self, values):
        if len(values) != ARRAY_GROUPS:
            raise ValueError("expected exactly two group values")
        result = fp32_add(values[0], values[1])
        return result.output_bits, 0, bool(result.invalid)

    def counters(self):
        group0_active = sum(cell.active_cycles for cell in self.groups[0].cells)
        group1_active = sum(cell.active_cycles for cell in self.groups[1].cells)
        group0_idle = sum(cell.idle_cycles for cell in self.groups[0].cells)
        group1_idle = sum(cell.idle_cycles for cell in self.groups[1].cells)
        return {
            "cell_count": self.cell_count(),
            "command_count": self.command_count,
            "mode_switch_count": self.mode_switch_count,
            "active_cycles": self.active_cycles,
            "idle_cycles": self.idle_cycles,
            "group0_active_cycles": group0_active,
            "group1_active_cycles": group1_active,
            "group0_idle_cycles": group0_idle,
            "group1_idle_cycles": group1_idle,
        }
