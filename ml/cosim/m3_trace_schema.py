"""Shared constants and helpers for ML-M3 real-weight RTL co-simulation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


MODEL_REPO = Path("D:/IC_Workspace/VEDA_ml_m2")
HARDWARE_REPO = Path("D:/IC_Workspace/VEDA")
ARTIFACT_ROOT = Path("D:/IC_Workspace/VEDA_artifacts/ml_m3")
Q2_BENCHMARK_ROOT = Path("D:/IC_Workspace/VEDA_artifacts/ml_q2/benchmark")
Q2_CHECKPOINT = Q2_BENCHMARK_ROOT / "checkpoints" / "VEDA-HWLM-1L64-Q2.pt"
Q2_EXPORT_DIR = Q2_BENCHMARK_ROOT / "exports"
Q2_TRACE_DIR = Q2_BENCHMARK_ROOT / "traces"
Q2_TOKENIZER = Path("D:/IC_Workspace/VEDA_artifacts/ml_m2/formal/tokenizer/tokenizer.json")
Q2_CHECKPOINT_SHA256 = "68b520f1322c79e568c39115809b8d623e21478af1662658cf997bf174cc9214"
Q2_TOKENIZER_SHA256 = "72c4100b9c923f8fc89ea563cdf18743742b87ad7cda6732606b61f50f290a1a"
H9_ACCEPTED_TAG = "hw-h9-real-weight-numeric-repair-accepted"
H9_ACCEPTED_COMMIT = "a54e608a8dc7e63c7e5dd342f8b893bb1e0b7485"

VECTOR_LENGTHS_REQUIRED = [1, 2, 8, 16]
VECTOR_LENGTHS_EXTENDED = [32]
NEXT_TOKEN_LENGTHS = [1, 8, 16]

MODEL_CONFIG_EXPECTED = {
    "vocab_size": 2048,
    "context_length": 128,
    "num_layers": 1,
    "d_model": 64,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "d_head": 8,
    "d_ffn": 256,
    "activation": "relu",
    "bias": False,
    "dropout": 0.0,
    "tie_word_embeddings": True,
}

RTL_WEIGHT_KINDS = {
    "wq": 0,
    "wk": 1,
    "wv": 2,
    "wo": 3,
    "norm1_gamma": 4,
    "norm2_gamma": 5,
    "w1": 6,
    "w2": 7,
}

RTL_WEIGHT_STATE = {
    "wq": "layers.0.attn.wq.weight",
    "wk": "layers.0.attn.wk.weight",
    "wv": "layers.0.attn.wv.weight",
    "wo": "layers.0.attn.wo.weight",
    "norm1_gamma": "layers.0.norm1.weight",
    "norm2_gamma": "layers.0.norm2.weight",
    "w1": "layers.0.ffn.w1.weight",
    "w2": "layers.0.ffn.w2.weight",
}

RTL_WEIGHT_SHAPES = {
    "wq": [64, 64],
    "wk": [64, 64],
    "wv": [64, 64],
    "wo": [64, 64],
    "norm1_gamma": [64],
    "norm2_gamma": [64],
    "w1": [256, 64],
    "w2": [64, 256],
}

EXPORT_TENSORS = [
    "token_embedding",
    "position_embedding",
    "final_norm_gamma",
    "lm_head",
    "norm1_gamma",
    "wq",
    "wk",
    "wv",
    "wo",
    "norm2_gamma",
    "w1",
    "w2",
]


def ensure_artifact_dirs() -> None:
    for name in ["manifests", "vectors", "traces", "rtl_logs", "waveforms", "comparisons", "temporary_build"]:
        (ARTIFACT_ROOT / name).mkdir(parents=True, exist_ok=True)


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def repo_relative(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(MODEL_REPO)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")
