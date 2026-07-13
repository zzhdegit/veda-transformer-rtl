import json
from pathlib import Path

from ml.training.smoke import run_smoke_training
from ml.training.train import load_stage_config


def test_cpu_smoke_training_loss_reload_generation_and_incremental(tmp_path: Path):
    config = load_stage_config("ml/configs/ml_m2_smoke.json")
    config["training"]["steps"] = 8
    config["model"]["context_length"] = 32
    result = run_smoke_training(tmp_path, config)
    metrics = result["metrics"]
    assert metrics.no_nan_inf
    assert metrics.final_train_loss < metrics.initial_train_loss
    assert result["reload_max_abs_diff"] == 0.0
    assert result["incremental_compare"]["allclose"]
    assert Path(result["checkpoint_manifest"]["path"]).exists()
    assert len(result["checkpoint_manifest"]["sha256"]) == 64
    generation = json.loads(Path(result["generation_path"]).read_text(encoding="utf-8"))
    assert generation["generated"]

