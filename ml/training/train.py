"""Command-line training entry point for ML-M2."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ml.architecture.config import HardwareMatchedConfig
from ml.data.dataset_manifest import artifact_root
from ml.training.formal_train import FormalTrainingConfig, run_formal_training
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
        training = data.get("training", {})
        cfg = FormalTrainingConfig(
            batch_size=int(training.get("batch_size", 512)),
            epochs=int(training.get("epochs", 3)),
            validation_interval=int(training.get("eval_interval", 100)),
            learning_rate=float(training.get("learning_rate", 3.0e-4)),
            weight_decay=float(training.get("weight_decay", 0.1)),
            grad_clip_norm=float(training.get("grad_clip_norm", 1.0)),
            seed=int(training.get("seed", 20260713)),
        )
        result = run_formal_training(args.output_dir, cfg)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    result = run_smoke_training(Path(args.output_dir), data)
    metrics = asdict(result["metrics"])
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"checkpoint={result['checkpoint_manifest']['path']}")
    print(f"checkpoint_sha256={result['checkpoint_manifest']['sha256']}")


if __name__ == "__main__":
    main()
