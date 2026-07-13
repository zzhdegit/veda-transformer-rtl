"""Stage 7 two-layer FFN reference."""

from collections import namedtuple

from model.projection.gemv_reference import gemv, output_row_major_address
from model.transformer.relu_reference import relu_quantize


FfnTrace = namedtuple("FfnTrace", ["ffn1_fp32", "relu_fp32", "activation_fp16", "ffn2_fp32", "invalid", "w1", "relu", "w2"])


def ffn_forward(norm2_fp16, w1, w2, pe_num):
    if not w1 or not w2:
        raise ValueError("FFN weights must be non-empty")
    d_model = len(norm2_fp16)
    d_ffn = len(w1)
    if any(len(row) != d_model for row in w1):
        raise ValueError("W1 row length must equal D_MODEL")
    if len(w2) != d_model:
        raise ValueError("W2 output rows must equal D_MODEL")
    if any(len(row) != d_ffn for row in w2):
        raise ValueError("W2 row length must equal D_FFN")

    w1_trace = gemv(norm2_fp16, w1, pe_num)
    relu_trace = relu_quantize(w1_trace.outputs)
    w2_trace = gemv(relu_trace.activation_fp16, w2, pe_num)
    invalid = any(row.invalid for row in w1_trace.rows) or relu_trace.invalid or any(row.invalid for row in w2_trace.rows)
    return FfnTrace(
        ffn1_fp32=w1_trace.outputs,
        relu_fp32=relu_trace.relu_fp32,
        activation_fp16=relu_trace.activation_fp16,
        ffn2_fp32=w2_trace.outputs,
        invalid=invalid,
        w1=w1_trace,
        relu=relu_trace,
        w2=w2_trace,
    )


__all__ = ["FfnTrace", "ffn_forward", "output_row_major_address"]
