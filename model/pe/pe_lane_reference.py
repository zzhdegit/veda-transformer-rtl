"""Stage 2 PE lane bit model."""

from model.arithmetic.fp32_mac_reference import fp32_mac


PE_LANE_MODE_PRODUCT = 0
PE_LANE_MODE_FMA = 1


def pe_lane_compute(mode, scalar_fp32, vector_fp32, accumulator_fp32=0, lane_active=True):
    if mode == PE_LANE_MODE_PRODUCT:
        a = scalar_fp32 if lane_active else 0
        b = vector_fp32 if lane_active else 0
        c = 0
    elif mode == PE_LANE_MODE_FMA:
        a = scalar_fp32 if lane_active else 0
        b = vector_fp32 if lane_active else 0
        c = accumulator_fp32
    else:
        raise ValueError("unsupported PE lane mode")
    return fp32_mac(a, b, c, "fused")
