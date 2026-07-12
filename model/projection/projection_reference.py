"""Stage 6 Q/K/V and output projection reference helpers."""

from collections import namedtuple

from model.projection.fp32_fp16_reference import fp32_to_fp16_bits
from model.projection.gemv_reference import gemv


WQ = 0
WK = 1
WV = 2
WO = 3

MATRIX_KIND_NAMES = {
    WQ: "WQ",
    WK: "WK",
    WV: "WV",
    WO: "WO",
}

QKVProjectionTrace = namedtuple(
    "QKVProjectionTrace",
    [
        "q_fp32",
        "k_fp32",
        "v_fp32",
        "q_fp16",
        "k_fp16",
        "v_fp16",
        "q_heads",
        "k_heads",
        "v_heads",
        "q_trace",
        "k_trace",
        "v_trace",
        "quantization",
        "qkv_stream",
    ],
)

OutputProjectionTrace = namedtuple(
    "OutputProjectionTrace",
    [
        "concat_fp32",
        "concat_fp16",
        "quantization",
        "output_fp32",
        "gemv_trace",
    ],
)


def validate_config(n_head, d_head, d_model):
    if n_head <= 0 or d_head <= 0 or d_model <= 0:
        raise ValueError("parameters must be positive")
    if d_model != n_head * d_head:
        raise ValueError("D_MODEL must equal N_HEAD * D_HEAD")


def projection_output_index(head, dim, d_head):
    if head < 0 or dim < 0 or dim >= d_head:
        raise ValueError("head/dim out of range")
    return head * d_head + dim


def split_projection_heads(vector, n_head, d_head):
    if len(vector) != n_head * d_head:
        raise ValueError("vector length does not match N_HEAD * D_HEAD")
    return [
        [vector[projection_output_index(head, dim, d_head)] for dim in range(d_head)]
        for head in range(n_head)
    ]


def concat_attention_heads(head_outputs_fp32, n_head, d_head):
    if len(head_outputs_fp32) != n_head:
        raise ValueError("head output count mismatch")
    concat = []
    for head in range(n_head):
        if len(head_outputs_fp32[head]) != d_head:
            raise ValueError("head output dimension mismatch")
        for dim in range(d_head):
            concat.append(head_outputs_fp32[head][dim])
    return concat


def qkv_stream_from_heads(q_heads, k_heads, v_heads, n_head, d_head):
    stream = []
    for head in range(n_head):
        for dim in range(d_head):
            stream.append(
                {
                    "head": head,
                    "dim": dim,
                    "q_fp16": q_heads[head][dim],
                    "k_fp16": k_heads[head][dim],
                    "v_fp16": v_heads[head][dim],
                    "last_dim": dim == d_head - 1,
                    "last_head": head == n_head - 1 and dim == d_head - 1,
                }
            )
    return stream


def qkv_projection(hidden_fp16, weights, n_head, d_head, pe_num):
    d_model = n_head * d_head
    validate_config(n_head, d_head, d_model)
    if len(hidden_fp16) != d_model:
        raise ValueError("hidden length mismatch")
    for kind in (WQ, WK, WV):
        if kind not in weights:
            raise ValueError("missing %s weights" % MATRIX_KIND_NAMES[kind])
        if len(weights[kind]) != d_model:
            raise ValueError("%s output row count mismatch" % MATRIX_KIND_NAMES[kind])

    q_trace = gemv(hidden_fp16, weights[WQ], pe_num)
    k_trace = gemv(hidden_fp16, weights[WK], pe_num)
    v_trace = gemv(hidden_fp16, weights[WV], pe_num)

    q_quant = [fp32_to_fp16_bits(value) for value in q_trace.outputs]
    k_quant = [fp32_to_fp16_bits(value) for value in k_trace.outputs]
    v_quant = [fp32_to_fp16_bits(value) for value in v_trace.outputs]
    q_fp16 = [result.output_bits for result in q_quant]
    k_fp16 = [result.output_bits for result in k_quant]
    v_fp16 = [result.output_bits for result in v_quant]
    q_heads = split_projection_heads(q_fp16, n_head, d_head)
    k_heads = split_projection_heads(k_fp16, n_head, d_head)
    v_heads = split_projection_heads(v_fp16, n_head, d_head)

    return QKVProjectionTrace(
        q_fp32=q_trace.outputs,
        k_fp32=k_trace.outputs,
        v_fp32=v_trace.outputs,
        q_fp16=q_fp16,
        k_fp16=k_fp16,
        v_fp16=v_fp16,
        q_heads=q_heads,
        k_heads=k_heads,
        v_heads=v_heads,
        q_trace=q_trace,
        k_trace=k_trace,
        v_trace=v_trace,
        quantization={"q": q_quant, "k": k_quant, "v": v_quant},
        qkv_stream=qkv_stream_from_heads(q_heads, k_heads, v_heads, n_head, d_head),
    )


def output_projection(head_outputs_fp32, w_o, n_head, d_head, pe_num):
    d_model = n_head * d_head
    validate_config(n_head, d_head, d_model)
    if len(w_o) != d_model:
        raise ValueError("W_O output row count mismatch")
    concat_fp32 = concat_attention_heads(head_outputs_fp32, n_head, d_head)
    quant = [fp32_to_fp16_bits(value) for value in concat_fp32]
    concat_fp16 = [result.output_bits for result in quant]
    gemv_trace = gemv(concat_fp16, w_o, pe_num)
    return OutputProjectionTrace(
        concat_fp32=concat_fp32,
        concat_fp16=concat_fp16,
        quantization=quant,
        output_fp32=gemv_trace.outputs,
        gemv_trace=gemv_trace,
    )
