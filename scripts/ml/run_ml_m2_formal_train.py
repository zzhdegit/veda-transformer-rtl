"""Run ML-M2 Formal benchmark and TinyStories training."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.training.formal_train import FormalTrainingConfig, benchmark_batches, run_formal_training


def main() -> int:
    batch_size = int(os.environ.get("VEDA_ML_M2_BATCH_SIZE", "512"))
    epochs = int(os.environ.get("VEDA_ML_M2_EPOCHS", "3"))
    validation_interval = int(os.environ.get("VEDA_ML_M2_VALIDATION_INTERVAL", "100"))
    benchmark = benchmark_batches()
    print(json.dumps(benchmark, indent=2, sort_keys=True))
    selected = benchmark.get("selected_batch_size") or batch_size
    cfg = FormalTrainingConfig(
        batch_size=int(os.environ.get("VEDA_ML_M2_BATCH_SIZE", str(selected))),
        epochs=epochs,
        validation_interval=validation_interval,
    )
    metrics = run_formal_training(cfg=cfg)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
