"""Export manifest data structures."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TensorExportRecord:
    logical_name: str
    source_state_dict_name: str
    shape: list[int]
    dtype: str
    source_layout: str
    rtl_layout: str
    transpose_applied: bool
    element_count: int
    sha256: str
    output_path: str


@dataclass(frozen=True)
class ExportManifest:
    stage: str
    tensor_count: int
    records: list[TensorExportRecord]


def write_export_manifest(manifest: ExportManifest, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_export_manifest(path: str | Path) -> ExportManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ExportManifest(
        stage=data["stage"],
        tensor_count=int(data["tensor_count"]),
        records=[TensorExportRecord(**record) for record in data["records"]],
    )

