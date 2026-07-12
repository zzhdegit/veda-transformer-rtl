"""Bit-accurate Stage 1B FP16-to-FP32 conversion reference.

The first Stage 1B numeric policy supports FP16 normal finite values and signed
zero. FP16 subnormals flush to signed zero. FP16 NaN/Inf are illegal inputs but
still produce a defined signed-zero output for deterministic simulation.
"""


FP16_EXP_W = 5
FP16_FRAC_W = 10
FP32_EXP_W = 8
FP32_FRAC_W = 23

FP16_EXP_BIAS = 15
FP32_EXP_BIAS = 127


def _mask(width):
    return (1 << width) - 1


def classify_fp16(bits):
    bits &= 0xFFFF
    exp = (bits >> FP16_FRAC_W) & _mask(FP16_EXP_W)
    frac = bits & _mask(FP16_FRAC_W)
    if exp == 0:
        return "zero" if frac == 0 else "subnormal"
    if exp == _mask(FP16_EXP_W):
        return "inf" if frac == 0 else "nan"
    return "normal"


def fp16_to_fp32_bits(bits):
    bits &= 0xFFFF
    sign = (bits >> 15) & 0x1
    exp16 = (bits >> FP16_FRAC_W) & _mask(FP16_EXP_W)
    frac16 = bits & _mask(FP16_FRAC_W)
    category = classify_fp16(bits)

    invalid = category in ("inf", "nan")
    underflow_or_ftz = category == "subnormal"

    if category == "normal":
        exp32 = exp16 - FP16_EXP_BIAS + FP32_EXP_BIAS
        out_bits = (sign << 31) | (exp32 << FP32_FRAC_W) | (frac16 << (FP32_FRAC_W - FP16_FRAC_W))
    else:
        out_bits = sign << 31

    return {
        "input_bits": bits,
        "output_bits": out_bits & 0xFFFFFFFF,
        "category": category,
        "invalid": invalid,
        "underflow_or_ftz": underflow_or_ftz,
    }


def fp16_to_fp32_tuple(bits):
    result = fp16_to_fp32_bits(bits)
    return (
        result["output_bits"],
        bool(result["invalid"]),
        bool(result["underflow_or_ftz"]),
        result["category"],
    )
