"""Manifest helpers for datasets used by ML-M2."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any


ACCESS_DATE = "2026-07-13"


def env_path(name: str, fallback: str) -> Path:
    return Path(os.environ.get(name, fallback)).expanduser()


def data_root() -> Path:
    return env_path("VEDA_ML_DATA_ROOT", "build/ml_m2_artifacts/datasets")


def artifact_root() -> Path:
    return env_path("VEDA_ML_ARTIFACT_ROOT", "build/ml_m2_artifacts")


@dataclass(frozen=True)
class DatasetManifest:
    name: str
    source_type: str
    source_url: str
    revision: str
    access_date: str
    license: str
    local_path: str | None = None
    sha256: str | None = None
    num_documents: int | None = None
    num_characters: int | None = None
    notes: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_manifest(manifest: DatasetManifest, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_manifest(path: str | Path) -> DatasetManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return DatasetManifest(**data)


def environment_manifest() -> dict[str, str]:
    return {
        "access_date": ACCESS_DATE,
        "veda_ml_data_root": str(data_root()),
        "veda_ml_artifact_root": str(artifact_root()),
        "veda_hf_cache": os.environ.get("VEDA_HF_CACHE", ""),
        "current_date": date.today().isoformat(),
    }

