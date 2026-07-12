"""Stage 6 shared GEMV projection reference."""

from collections import namedtuple

from model.pe.pe_core_reference import MODE_GEMV, inner_product_tiles


GemvOutputTrace = namedtuple("GemvOutputTrace", ["output_index", "output_bits", "tiles", "invalid"])
GemvTrace = namedtuple("GemvTrace", ["outputs", "rows"])


def output_row_major_address(output_index, input_index, input_length):
    if output_index < 0 or input_index < 0 or input_index >= input_length:
        raise ValueError("weight address out of range")
    return output_index * input_length + input_index


def gemv_output_row(input_fp16, weight_row_fp16, pe_num, output_index=0):
    if len(input_fp16) != len(weight_row_fp16):
        raise ValueError("input and weight row length mismatch")
    result, tiles = inner_product_tiles(input_fp16, weight_row_fp16, pe_num, MODE_GEMV)
    invalid = any(bool(tile["invalid"]) for tile in tiles)
    return GemvOutputTrace(output_index, result, tiles, invalid)


def gemv(input_fp16, weights_output_row_major, pe_num):
    if not weights_output_row_major:
        raise ValueError("weights must contain at least one output row")
    rows = []
    outputs = []
    for output_index, row in enumerate(weights_output_row_major):
        trace = gemv_output_row(input_fp16, row, pe_num, output_index)
        rows.append(trace)
        outputs.append(trace.output_bits)
    return GemvTrace(outputs, rows)
