"""Command-line training entry point for ML-M2."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ml.architecture.config import HardwareMatchedConfig
from ml.data.dataset_manifest import artifact_root
from ml.training.smoke import run_smoke_training


def load_stage_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def model_config_from_stage_config(data: dict) -> HardwareMatchedConfig:
    model = dict(data["model"])
    seed = int(data.get("training", {}).get("seed", 20260713))
    model["seed"] = seed
    return HardwareMatchedConfig(**model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ml/configs/ml_m2_smoke.json")
    parser.add_argument("--output-dir", default=str(artifact_root() / "smoke"))
    parser.add_argument("--mode", choices=["smoke", "formal"], default="smoke")
    args = parser.parse_args()

    data = load_stage_config(args.config)
    if args.mode == "formal":
        raise SystemExit("formal TinyStories training workflow is defined in ML-M2E and requires a GPU artifact setup")
    result = run_smoke_training(Path(args.output_dir), data)
    metrics = asdict(result["metrics"])
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"checkpoint={result['checkpoint_manifest']['path']}")
    print(f"checkpoint_sha256={result['checkpoint_manifest']['sha256']}")


if __name__ == "__main__":
    main()

