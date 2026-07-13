"""Stage 7 RMSNorm bit-model helpers."""

from collections import namedtuple

from model.arithmetic.fp16_fp32_reference import fp16_to_fp32_bits
from model.arithmetic.fp32_add_reference import fp32_add
from model.arithmetic.fp32_mac_reference import (
    classify_fp32,
    fp32_bits_to_fraction,
    fp32_mac,
    fraction_to_fp32_bits,
    is_supported_fp32_operand,
)
from model.attention.softmax_reference import float_to_fp32_bits, fp32_recip, fp32_to_float
from model.projection.fp32_fp16_reference import fp32_to_fp16_bits


FP32_ZERO = 0x00000000
FP32_ONE = 0x3F800000
EPS_FP32_DEFAULT = 0x3727C5AC

RmsNormTrace = namedtuple(
    "RmsNormTrace",
    [
        "input_fp32",
        "gamma_fp32",
        "sum_sq",
        "sum_trace",
        "mean_scale",
        "mean_sq",
        "mean_sq_eps",
        "sqrt",
        "inv_rms",
        "norm_fp32",
        "norm_fp16",
        "invalid",
    ],
)


def is_power_of_two(value):
    return value > 0 and (value & (value - 1)) == 0


def fp32_mul(a_bits, b_bits):
    return fp32_mac(a_bits, b_bits, FP32_ZERO, "fused")


def fp32_sqrt(bits):
    bits &= 0xFFFFFFFF
    if not is_supported_fp32_operand(bits):
        return FP32_ZERO, True
    value = fp32_bits_to_fraction(bits)
    if value < 0:
        return FP32_ZERO, True
    return float_to_fp32_bits(fp32_to_float(bits) ** 0.5), False


def fp32_recip_checked(bits):
    bits &= 0xFFFFFFFF
    invalid = (not is_supported_fp32_operand(bits)) or fp32_bits_to_fraction(bits) == 0
    if invalid:
        return FP32_ZERO, True
    return fp32_recip(bits), False


def mean_scale_for_d_model(d_model):
    if not is_power_of_two(d_model):
        raise ValueError("D_MODEL must be a power of two")
    return fraction_to_fp32_bits(fp32_bits_to_fraction(FP32_ONE) / d_model)


def fp16_vector_to_fp32(values):
    out = []
    invalid = False
    for value in values:
        conv = fp16_to_fp32_bits(value)
        out.append(conv["output_bits"])
        invalid = invalid or bool(conv["invalid"])
    return out, invalid


def rmsnorm(input_fp32, gamma_fp16, eps_fp32=EPS_FP32_DEFAULT):
    if len(input_fp32) != len(gamma_fp16):
        raise ValueError("RMSNorm input/gamma length mismatch")
    d_model = len(input_fp32)
    if not is_power_of_two(d_model):
        raise ValueError("D_MODEL must be a power of two")

    gamma_fp32, gamma_invalid = fp16_vector_to_fp32(gamma_fp16)
    invalid = bool(gamma_invalid)
    acc = FP32_ZERO
    sum_trace = []
    for dim, value in enumerate(input_fp32):
        mac = fp32_mac(value, value, acc, "fused")
        acc = mac.output_bits
        invalid = invalid or bool(mac.invalid)
        sum_trace.append({"dim": dim, "input": value & 0xFFFFFFFF, "acc": acc, "invalid": bool(mac.invalid)})

    mean_scale = mean_scale_for_d_model(d_model)
    mean_mul = fp32_mul(acc, mean_scale)
    mean_sq = mean_mul.output_bits
    invalid = invalid or bool(mean_mul.invalid)

    eps_add = fp32_add(mean_sq, eps_fp32)
    mean_sq_eps = eps_add.output_bits
    invalid = invalid or bool(eps_add.invalid)

    sqrt_bits, sqrt_invalid = fp32_sqrt(mean_sq_eps)
    invalid = invalid or bool(sqrt_invalid)
    inv_rms, recip_invalid = fp32_recip_checked(sqrt_bits)
    invalid = invalid or bool(recip_invalid)

    norm_fp32 = []
    norm_fp16 = []
    for value, gamma in zip(input_fp32, gamma_fp32):
        first_mul = fp32_mul(value, inv_rms)
        second_mul = fp32_mul(first_mul.output_bits, gamma)
        quant = fp32_to_fp16_bits(second_mul.output_bits)
        invalid = invalid or bool(first_mul.invalid) or bool(second_mul.invalid) or bool(quant.invalid)
        norm_fp32.append(second_mul.output_bits)
        norm_fp16.append(quant.output_bits)

    if classify_fp32(eps_fp32) in ("inf", "nan"):
        invalid = True

    return RmsNormTrace(
        input_fp32=[value & 0xFFFFFFFF for value in input_fp32],
        gamma_fp32=gamma_fp32,
        sum_sq=acc,
        sum_trace=sum_trace,
        mean_scale=mean_scale,
        mean_sq=mean_sq,
        mean_sq_eps=mean_sq_eps,
        sqrt=sqrt_bits,
        inv_rms=inv_rms,
        norm_fp32=norm_fp32,
        norm_fp16=norm_fp16,
        invalid=invalid,
    )
