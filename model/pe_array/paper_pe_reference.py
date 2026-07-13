"""Reference PE cell for the Stage 8 paper-structured array."""

from collections import namedtuple

from model.arithmetic.fp16_fp32_reference import fp16_to_fp32_bits
from model.pe.pe_lane_reference import PE_LANE_MODE_FMA, PE_LANE_MODE_PRODUCT, pe_lane_compute
from model.pe_array.paper_array_mapping import PE_TYPE_A, PE_TYPE_B, pe_type_for_column


PaperPEOutput = namedtuple(
    "PaperPEOutput",
    [
        "group",
        "row",
        "column",
        "pe_type",
        "active",
        "product",
        "accumulator",
        "forwarded_partial",
        "status",
        "invalid",
    ],
)


class PaperPECell(object):
    """One physical PE cell with stable row, column, group, and type identity."""

    def __init__(self, group, row, column):
        self.group = group
        self.row = row
        self.column = column
        self.pe_type = pe_type_for_column(column)
        self.accumulator = 0
        self.forwarded_partial = 0
        self.status = 0
        self.invalid = False
        self.active_cycles = 0
        self.idle_cycles = 0

    @property
    def is_type_a(self):
        return self.pe_type == PE_TYPE_A

    @property
    def is_type_b(self):
        return self.pe_type == PE_TYPE_B

    def reset(self):
        self.accumulator = 0
        self.forwarded_partial = 0
        self.status = 0
        self.invalid = False
        self.active_cycles = 0
        self.idle_cycles = 0

    def product(self, operand_a_fp16, operand_b_fp16, active=True):
        if active:
            a32 = fp16_to_fp32_bits(operand_a_fp16)["output_bits"]
            b32 = fp16_to_fp32_bits(operand_b_fp16)["output_bits"]
            lane_result = pe_lane_compute(PE_LANE_MODE_PRODUCT, a32, b32, 0, True)
            self.forwarded_partial = lane_result.output_bits
            self.status = 0
            self.invalid = bool(lane_result.invalid)
            self.active_cycles += 1
            return PaperPEOutput(
                self.group,
                self.row,
                self.column,
                self.pe_type,
                True,
                lane_result.output_bits,
                self.accumulator,
                self.forwarded_partial,
                self.status,
                self.invalid,
            )

        self.forwarded_partial = 0
        self.status = 0
        self.invalid = False
        self.idle_cycles += 1
        return PaperPEOutput(
            self.group,
            self.row,
            self.column,
            self.pe_type,
            False,
            0,
            self.accumulator,
            self.forwarded_partial,
            0,
            False,
        )

    def outer_accumulate(self, scalar_fp32, operand_b_fp16, active=True):
        if active:
            b32 = fp16_to_fp32_bits(operand_b_fp16)["output_bits"]
            lane_result = pe_lane_compute(PE_LANE_MODE_FMA, scalar_fp32, b32, self.accumulator, True)
            self.accumulator = lane_result.output_bits
            self.forwarded_partial = self.accumulator
            self.status = 0
            self.invalid = bool(lane_result.invalid)
            self.active_cycles += 1
        else:
            self.status = 0
            self.invalid = False
            self.idle_cycles += 1

        return PaperPEOutput(
            self.group,
            self.row,
            self.column,
            self.pe_type,
            bool(active),
            0,
            self.accumulator,
            self.forwarded_partial,
            self.status,
            self.invalid,
        )


def expected_type_a_columns():
    return [0, 2, 4, 6]


def expected_type_b_columns():
    return [1, 3, 5, 7]


def type_mapping_legal(cells):
    for cell in cells:
        expected = PE_TYPE_A if cell.column in expected_type_a_columns() else PE_TYPE_B
        if cell.pe_type != expected:
            return False
    return True
