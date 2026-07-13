"""Run the complete ML-M2 pipeline test suite."""

from __future__ import annotations

import subprocess
import sys


PY_COMPILE_FILES = [
    "ml/data/dataset_manifest.py",
    "ml/data/tinystories_loader.py",
    "ml/data/tiny_shakespeare_loader.py",
    "ml/data/sequence_builder.py",
    "ml/data/dataset_hash.py",
    "ml/data/formal_data.py",
    "ml/tokenizer/train_bpe.py",
    "ml/tokenizer/load_tokenizer.py",
    "ml/tokenizer/validate_tokenizer.py",
    "ml/tokenizer/tokenizer_manifest.py",
    "ml/architecture/config.py",
    "ml/architecture/rmsnorm.py",
    "ml/architecture/attention.py",
    "ml/architecture/feed_forward.py",
    "ml/architecture/transformer_layer.py",
    "ml/architecture/causal_lm.py",
    "ml/architecture/state_dict_mapping.py",
    "ml/training/reproducibility.py",
    "ml/training/optimizer.py",
    "ml/training/scheduler.py",
    "ml/training/checkpoint.py",
    "ml/training/trainer.py",
    "ml/training/train.py",
    "ml/training/smoke.py",
    "ml/training/formal.py",
    "ml/training/formal_train.py",
    "ml/training/numeric_audit.py",
    "ml/inference/generate.py",
    "ml/inference/incremental_decode.py",
    "ml/inference/prompt_suite.py",
    "ml/export/export_manifest.py",
    "ml/export/export_rtl_streams.py",
    "ml/export/export_fp16_weights.py",
    "ml/export/validate_export.py",
    "ml/export/export_checkpoint.py",
    "ml/export/export_trace.py",
    "ml/export/formal_export.py",
    "ml/cosim/fp16_policy.py",
    "ml/cosim/hardware_aware_layer.py",
    "ml/cosim/hardware_aware_model.py",
    "ml/cosim/trace_compare.py",
    "ml/cosim/rtl_vector_builder.py",
    "ml/evaluation/evaluate_loss.py",
    "ml/evaluation/evaluate_perplexity.py",
    "ml/evaluation/evaluate_generation.py",
    "ml/evaluation/evaluate_quantization.py",
    "ml/evaluation/compare_checkpoints.py",
    "scripts/ml/run_ml_m2_data_tests.py",
    "scripts/ml/run_ml_m2_unit_tests.py",
    "scripts/ml/run_ml_m2_smoke_test.py",
    "scripts/ml/run_ml_m2_export_trace_tests.py",
    "scripts/ml/run_ml_m2_gpu_check.py",
    "scripts/ml/run_ml_m2_numeric_audit.py",
    "scripts/ml/run_ml_m2_formal_data.py",
    "scripts/ml/run_ml_m2_formal_train.py",
    "scripts/ml/run_ml_m2_formal_eval.py",
    "scripts/ml/run_ml_m2_formal_export.py",
]


def run(cmd: list[str]) -> int:
    print("$ " + " ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    commands = [
        [sys.executable, "-m", "py_compile", *PY_COMPILE_FILES],
        [sys.executable, "scripts/ml/run_ml_m2_data_tests.py"],
        [sys.executable, "scripts/ml/run_ml_m2_unit_tests.py"],
        [sys.executable, "scripts/ml/run_ml_m2_smoke_test.py"],
        [sys.executable, "scripts/ml/run_ml_m2_export_trace_tests.py"],
        [sys.executable, "-m", "pytest", "ml/tests/test_training_numerics.py"],
    ]
    for cmd in commands:
        code = run(cmd)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
