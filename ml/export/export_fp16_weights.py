"""Export ML-M2 tensors as FP16 `.npy` and `.hex` streams."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ml.architecture.state_dict_mapping import RTL_TENSOR_MAP, SOFTWARE_TENSOR_MAP
from ml.data.dataset_hash import sha256_file
from ml.export.export_manifest import ExportManifest, TensorExportRecord, write_export_manifest
from ml.export.export_rtl_streams import write_fp16_hex_stream


def _export_tensor(logical_name: str, source_name: str, tensor: torch.Tensor, output_dir: Path) -> TensorExportRecord:
    arr = tensor.detach().cpu().numpy().astype(np.float16)
    npy_path = output_dir / f"{logical_name}.npy"
    hex_path = output_dir / f"{logical_name}.hex"
    np.save(npy_path, arr)
    write_fp16_hex_stream(arr, hex_path)
    return TensorExportRecord(
        logical_name=logical_name,
        source_state_dict_name=source_name,
        shape=list(arr.shape),
        dtype="fp16",
        source_layout="[output_index][input_index]" if arr.ndim == 2 else "[index]",
        rtl_layout="weight[output_index][input_index]" if arr.ndim == 2 else "vector[index]",
        transpose_applied=False,
        element_count=int(arr.size),
        sha256=sha256_file(hex_path),
        output_path=str(hex_path),
    )


def export_fp16_weights(model, output_dir: str | Path) -> ExportManifest:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    records: list[TensorExportRecord] = []
    for logical, source in {**SOFTWARE_TENSOR_MAP, **RTL_TENSOR_MAP}.items():
        records.append(_export_tensor(logical, source, state[source], out))
    manifest = ExportManifest(stage="ML-M2F", tensor_count=len(records), records=records)
    write_export_manifest(manifest, out / "export_manifest.json")
    return manifest

