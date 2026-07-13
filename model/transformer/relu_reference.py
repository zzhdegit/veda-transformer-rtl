"""Stage 7 ReLU and activation quantization reference."""

from collections import namedtuple

from model.arithmetic.fp32_mac_reference import classify_fp32
from model.projection.fp32_fp16_reference import fp32_to_fp16_bits


FP32_ZERO = 0x00000000

ReluResult = namedtuple("ReluResult", ["output_bits", "invalid"])
ReluVectorTrace = namedtuple("ReluVectorTrace", ["relu_fp32", "activation_fp16", "invalid", "steps"])


def relu_fp32(bits):
    bits &= 0xFFFFFFFF
    category = classify_fp32(bits)
    if category in ("inf", "nan"):
        return ReluResult(FP32_ZERO, True)
    if (bits >> 31) & 1:
        return ReluResult(FP32_ZERO, False)
    if (bits & 0x7FFFFFFF) == 0:
        return ReluResult(FP32_ZERO, False)
    return ReluResult(bits, False)


def relu_quantize(values_fp32):
    relu_values = []
    activation = []
    steps = []
    invalid = False
    for dim, value in enumerate(values_fp32):
        relu = relu_fp32(value)
        quant = fp32_to_fp16_bits(relu.output_bits)
        invalid = invalid or bool(relu.invalid) or bool(quant.invalid)
        relu_values.append(relu.output_bits)
        activation.append(quant.output_bits)
        steps.append(
            {
                "dim": dim,
                "input": value & 0xFFFFFFFF,
                "relu": relu.output_bits,
                "fp16": quant.output_bits,
                "invalid": bool(relu.invalid) or bool(quant.invalid),
            }
        )
    return ReluVectorTrace(relu_values, activation, invalid, steps)
