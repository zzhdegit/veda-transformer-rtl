"""Formal TinyStories workflow planning for ML-M2."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ml.data.dataset_manifest import artifact_root, data_root, environment_manifest
from ml.training.reproducibility import environment_summary


def formal_training_status() -> dict:
    cuda_available = torch.cuda.is_available()
    return {
        "stage": "ML-M2E",
        "formal_training_status": "READY" if cuda_available else "PENDING",
        "pending_reason": "" if cuda_available else "CUDA GPU is not available in this environment.",
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else "",
        "data_root": str(data_root()),
        "artifact_root": str(artifact_root()),
        "required_dataset_files": [
            str(data_root() / "TinyStories-train.txt"),
            str(data_root() / "TinyStories-valid.txt"),
        ],
        "reproduction_command": (
            "python -m ml.training.train --mode formal "
            "--config ml/configs/ml_m2_formal.json "
            "--output-dir %VEDA_ML_ARTIFACT_ROOT%/formal"
        ),
        "environment": {**environment_manifest(), **environment_summary()},
    }


def write_formal_status(path: str | Path) -> dict:
    status = formal_training_status()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return status


def main() -> None:
    status = write_formal_status("reports/ml_m2/formal_training_status.json")
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

