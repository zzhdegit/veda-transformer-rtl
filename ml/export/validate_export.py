"""Validate ML-M2 export manifests and matrix layout."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from ml.data.dataset_hash import sha256_file
from ml.export.export_fp16_weights import _export_tensor
from ml.export.export_manifest import read_export_manifest


def validate_export_manifest(path: str | Path) -> dict:
    manifest = read_export_manifest(path)
    if manifest.tensor_count != len(manifest.records):
        raise ValueError("tensor_count mismatch")
    for record in manifest.records:
        if len(record.sha256) != 64:
            raise ValueError(f"bad sha256 for {record.logical_name}")
        if not Path(record.output_path).exists():
            raise ValueError(f"missing export file {record.output_path}")
        if sha256_file(record.output_path) != record.sha256:
            raise ValueError(f"sha mismatch for {record.logical_name}")
    return {"tensor_count": manifest.tensor_count}


def validate_linear_weight_direction(tmp_path: str | Path) -> list[str]:
    linear = nn.Linear(3, 2, bias=False)
    linear.weight.data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    record = _export_tensor("linear_direction", "linear.weight", linear.weight, Path(tmp_path))
    lines = Path(record.output_path).read_text(encoding="ascii").splitlines()
    return lines

