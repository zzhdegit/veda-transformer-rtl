"""Adapters around the accepted repository FP32/FP16 bit helpers."""

from __future__ import annotations

import torch

from model.arithmetic.fp16_fp32_reference import fp16_to_fp32_bits
from model.attention.softmax_reference import float_to_fp32_bits, fp32_to_float
from model.projection.fp32_fp16_reference import fp32_to_fp16_bits


def float_to_fp16_bits(value: float) -> int:
    return int(fp32_to_fp16_bits(float_to_fp32_bits(float(value))).output_bits)


def fp16_bits_to_float(bits: int) -> float:
    return fp32_to_float(fp16_to_fp32_bits(int(bits) & 0xFFFF)["output_bits"])


def fp32_bits_to_float(bits: int) -> float:
    return fp32_to_float(int(bits) & 0xFFFFFFFF)


def tensor_to_fp16_bits(tensor: torch.Tensor) -> list:
    values = tensor.detach().cpu().float().tolist()

    def convert(obj):
        if isinstance(obj, list):
            return [convert(item) for item in obj]
        return float_to_fp16_bits(float(obj))

    return convert(values)


def fp16_bits_nested_to_tensor(values) -> torch.Tensor:
    def convert(obj):
        if isinstance(obj, list):
            return [convert(item) for item in obj]
        return fp16_bits_to_float(int(obj))

    return torch.tensor(convert(values), dtype=torch.float32)

