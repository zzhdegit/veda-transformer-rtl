"""ML-Q2 full TinyStories hardware benchmark training workflow."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.request import Request, urlopen

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ml.architecture.causal_lm import HardwareMatchedCausalLM
from ml.architecture.config import HardwareMatchedConfig
from ml.cosim.fp16_policy import fp16_bits_nested_to_tensor
from ml.cosim.hardware_aware_model import run_hardware_aware_model
from ml.cosim.rtl_vector_builder import write_small_rtl_fixture
from ml.data.dataset_hash import sha256_file
from ml.data.formal_data import TINYSTORIES_SOURCE_COMMIT
from ml.data.tinystories_loader import (
    TINYSTORIES_CARD_URL,
    TINYSTORIES_LICENSE,
    TINYSTORIES_VALID_URL,
    split_tinystories_text,
)
from ml.evaluation.evaluate_quantization import logits_agreement, tensor_error_metrics
from ml.evaluation.q1_quality import (
    IGNORE_INDEX,
    ML_M2_BASELINE_CHECKPOINT,
    ML_M2_BASELINE_SHA256,
    _load_model_from_checkpoint,
    _load_packed_dataset,
    _pack_docs,
    _write_json,
    _write_md,
    evaluate_dataset_loss,
)
from ml.export.export_checkpoint import export_checkpoint
from ml.export.export_trace import export_trace
from ml.export.formal_export import _fp16_weight_rounded_model, _torch_load
from ml.export.validate_export import validate_export_manifest
from ml.inference.incremental_decode import compare_full_vs_incremental
from ml.inference.interactive import GenerationConfig, ModelBundle, generate_text_record, load_interactive_bundle
from ml.tokenizer.load_tokenizer import SimpleBPETokenizer
from ml.training.optimizer import build_optimizer
from ml.training.reproducibility import set_seed
from ml.training.scheduler import build_scheduler


ML_Q1_CHECKPOINT = Path("D:/IC_Workspace/VEDA_artifacts/ml_q1/candidate/checkpoints/ml_q1_candidate_best.pt")
ML_Q1_SHA256 = "fbb8b1815d03a0c9fb3cb1559c3cb6942038e7174d7d434ef57f79cb492994da"
ML_Q2_ROOT = Path("D:/IC_Workspace/VEDA_artifacts/ml_q2")
HF_TREE_URL = "https://huggingface.co/api/datasets/roneneldan/TinyStories/tree/main?recursive=true"
HF_DATASET_REVISION = TINYSTORIES_SOURCE_COMMIT
MODEL_NAME = "VEDA-HWLM-1L64-Q2"
HOLDOUT_GLOBAL_START = 300000
HOLDOUT_COUNT = 10000


@dataclass(frozen=True)
class Q2ExperimentConfig:
    name: str
    role: str
    init: str
    learning_rate: float
    minimum_lr: float
    warmup_fraction: float
    epochs: int
    max_epochs: int
    optional_threshold: float
    batch_size: int = 1024
    pilot_steps: int = 500
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1.0e-8
    grad_clip: float = 1.0
    seed: int = 20260713
    dtype: str = "bf16"


CONTINUATION_CFG = Q2ExperimentConfig(
    name="continuation",
    role="ML-Q2 continuation",
    init="ml_q1_best",
    learning_rate=5.0e-5,
    minimum_lr=5.0e-6,
    warmup_fraction=0.01,
    epochs=2,
    max_epochs=3,
    optional_threshold=0.005,
)

FROM_SCRATCH_CFG = Q2ExperimentConfig(
    name="from_scratch",
    role="ML-Q2 from-scratch",
    init="random",
    learning_rate=3.0e-4,
    minimum_lr=1.0e-5,
    warmup_fraction=0.02,
    epochs=3,
    max_epochs=4,
    optional_threshold=0.005,
)


def q2_root() -> Path:
    return Path(os.environ.get("VEDA_ML_Q2_ROOT", str(ML_Q2_ROOT)))


def _query_hf_tree() -> list[dict[str, Any]]:
    request = Request(HF_TREE_URL, headers={"User-Agent": "veda-ml-q2/1.0"})
    with urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def official_train_shards() -> list[dict[str, Any]]:
    cached = q2_root() / "data" / "full_dataset_manifest.json"
    if cached.exists():
        payload = json.loads(cached.read_text(encoding="utf-8"))
        if payload.get("official_train_shards"):
            return payload["official_train_shards"]
    entries = _query_hf_tree()
    shards = []
    for item in entries:
        path = str(item.get("path", ""))
        if path.startswith("data/train-") and path.endswith(".parquet"):
            shards.append(
                {
                    "path": path,
                    "filename": Path(path).name,
                    "url": f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/{path}",
                    "hf_oid": item.get("oid"),
                    "size_bytes": int(item.get("size", 0)),
                    "revision": HF_DATASET_REVISION,
                }
            )
    shards.sort(key=lambda row: row["path"])
    if not shards:
        raise RuntimeError("no official TinyStories train parquet shards found")
    return shards


def _download_range(url: str, part_path: Path, start: int, end: int, retries: int = 6) -> None:
    expected = end - start + 1
    if part_path.exists() and part_path.stat().st_size == expected:
        return
    if part_path.exists() and part_path.stat().st_size > expected:
        part_path.unlink()
    part_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = part_path.with_suffix(part_path.suffix + ".tmp")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = Request(
            url,
            headers={
                "User-Agent": "veda-ml-q2/1.0",
                "Range": f"bytes={start}-{end}",
            },
        )
        try:
            if tmp.exists():
                tmp.unlink()
            with urlopen(request, timeout=300) as response:
                status = getattr(response, "status", None)
                content_range = response.headers.get("Content-Range", "")
                if status != 206 or not content_range.startswith(f"bytes {start}-{end}/"):
                    raise RuntimeError(
                        f"range request failed for {start}-{end}: status={status} Content-Range={content_range!r}"
                    )
                with tmp.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
            if tmp.stat().st_size != expected:
                raise RuntimeError(f"range part {part_path} has {tmp.stat().st_size} bytes; expected {expected}")
            tmp.replace(part_path)
            return
        except Exception as exc:  # Network path: retry transient TLS/range failures.
            last_error = exc
            if tmp.exists():
                tmp.unlink()
            time.sleep(min(30, 2 * attempt))
    raise RuntimeError(f"failed to download byte range {start}-{end} after {retries} attempts") from last_error


def _download_with_ranges(url: str, path: Path, expected_size: int, workers: int = 12, chunk_size: int = 8 * 1024 * 1024) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == expected_size:
        return
    if path.exists() and path.stat().st_size > expected_size:
        raise RuntimeError(f"{path} is larger than expected; refusing to overwrite")
    existing_size = path.stat().st_size if path.exists() else 0
    part_dir = path.parent / f".{path.name}.parts"
    ranges = []
    offset = existing_size
    while offset < expected_size:
        end = min(offset + chunk_size - 1, expected_size - 1)
        ranges.append((offset, end, part_dir / f"{offset:012d}-{end:012d}.part"))
        offset = end + 1
    if ranges:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(ranges))) as pool:
            futures = [pool.submit(_download_range, url, part, start, end) for start, end, part in ranges]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        with path.open("ab") as handle:
            for start, end, part in ranges:
                if path.stat().st_size != start:
                    raise RuntimeError(f"{path} size changed during assembly: expected {start}, got {path.stat().st_size}")
                with part.open("rb") as part_handle:
                    shutil.copyfileobj(part_handle, handle, length=1024 * 1024)
    if path.stat().st_size != expected_size:
        raise RuntimeError(f"downloaded {path} has {path.stat().st_size} bytes; expected {expected_size}")
    shutil.rmtree(part_dir, ignore_errors=True)


def _ensure_shard(shard: dict[str, Any], target_dir: Path) -> dict[str, Any]:
    target = target_dir / shard["filename"]
    q1_cache = Path("D:/IC_Workspace/VEDA_artifacts/ml_q1/data/raw/parquet") / shard["filename"]
    if not target.exists() and q1_cache.exists() and q1_cache.stat().st_size == shard["size_bytes"]:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(q1_cache, target)
    start = time.perf_counter()
    _download_with_ranges(shard["url"], target, int(shard["size_bytes"]))
    return {
        **shard,
        "local_path": str(target),
        "sha256": sha256_file(target),
        "download_elapsed_seconds": time.perf_counter() - start,
    }


def load_tokenizer() -> SimpleBPETokenizer:
    bundle = load_interactive_bundle()
    return bundle.tokenizer


def _validate_frozen_inputs() -> dict[str, Any]:
    m2_sha = sha256_file(ML_M2_BASELINE_CHECKPOINT)
    q1_sha = sha256_file(ML_Q1_CHECKPOINT)
    if m2_sha != ML_M2_BASELINE_SHA256:
        raise RuntimeError(f"ML-M2 baseline SHA mismatch: {m2_sha}")
    if q1_sha != ML_Q1_SHA256:
        raise RuntimeError(f"ML-Q1 candidate SHA mismatch: {q1_sha}")
    q1_payload = _torch_load(ML_Q1_CHECKPOINT)
    cfg = HardwareMatchedConfig.from_json_dict(q1_payload["config"])
    cfg.validate()
    return {
        "ml_m2_baseline": str(ML_M2_BASELINE_CHECKPOINT),
        "ml_m2_sha256": m2_sha,
        "ml_q1_candidate": str(ML_Q1_CHECKPOINT),
        "ml_q1_sha256": q1_sha,
        "model_config": cfg.to_json_dict(),
    }


class _SequenceCounter:
    def __init__(self, tokenizer: SimpleBPETokenizer, context_length: int):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.buffer_len = 0
        self.sequence_count = 0
        self.token_count = 0
        self.eos_count = 0
        self.unk_count = 0

    def add(self, text: str) -> None:
        ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        self.token_count += len(ids)
        self.eos_count += ids.count(self.tokenizer.eos_id)
        self.unk_count += ids.count(self.tokenizer.unk_id)
        self.buffer_len += len(ids)
        need = self.context_length + 1
        while self.buffer_len >= need:
            self.sequence_count += 1
            self.buffer_len -= self.context_length

    def finalize(self) -> None:
        if self.buffer_len >= 2:
            self.sequence_count += 1


class _SequenceWriter:
    def __init__(self, tokenizer: SimpleBPETokenizer, context_length: int, sequence_count: int):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.inputs = torch.empty((sequence_count, context_length), dtype=torch.int16)
        self.labels = torch.empty((sequence_count, context_length), dtype=torch.int16)
        self.buffer: list[int] = []
        self.row = 0

    def _flush_full(self) -> None:
        need = self.context_length + 1
        while len(self.buffer) >= need:
            window = self.buffer[:need]
            del self.buffer[: self.context_length]
            self.inputs[self.row] = torch.tensor(window[:-1], dtype=torch.int16)
            self.labels[self.row] = torch.tensor(window[1:], dtype=torch.int16)
            self.row += 1

    def add(self, text: str) -> None:
        self.buffer.extend(self.tokenizer.encode(text, add_bos=True, add_eos=True))
        self._flush_full()

    def finalize(self) -> None:
        if len(self.buffer) >= 2:
            need = self.context_length + 1
            window = self.buffer[:need]
            if len(window) < need:
                window = window + [self.tokenizer.pad_id] * (need - len(window))
            labels = [token if token != self.tokenizer.pad_id else IGNORE_INDEX for token in window[1:]]
            self.inputs[self.row] = torch.tensor(window[:-1], dtype=torch.int16)
            self.labels[self.row] = torch.tensor(labels, dtype=torch.int16)
            self.row += 1
        if self.row != self.inputs.shape[0]:
            raise RuntimeError(f"packed row mismatch: wrote {self.row}, expected {self.inputs.shape[0]}")


def _iter_shard_texts(shard_records: list[dict[str, Any]]) -> Iterable[tuple[int, int, str]]:
    import pyarrow.parquet as pq

    global_index = 0
    for shard_index, shard in enumerate(shard_records):
        parquet = pq.ParquetFile(shard["local_path"])
        text_column = None
        for field in parquet.schema_arrow:
            if str(field.type) in {"string", "large_string"}:
                text_column = field.name
                break
        if text_column is None:
            raise RuntimeError(f"no text column found in {shard['local_path']}")
        shard["row_count"] = parquet.metadata.num_rows
        shard["text_column"] = text_column
        for batch in parquet.iter_batches(columns=[text_column], batch_size=8192):
            for value in batch.column(0).to_pylist():
                text = "" if value is None else str(value).strip()
                yield global_index, shard_index, text
                global_index += 1


def _pack_validation_holdout(tokenizer: SimpleBPETokenizer, context_length: int, out_dir: Path) -> dict[str, Any]:
    q1_raw = json.loads(Path("D:/IC_Workspace/VEDA_artifacts/ml_q1/data/q1_raw_data_manifest.json").read_text(encoding="utf-8"))
    result = {}
    for split in ["validation", "holdout"]:
        docs = split_tinystories_text(Path(q1_raw[split]["path"]).read_text(encoding="utf-8", errors="replace"))
        inputs, labels, stats = _pack_docs(docs[: q1_raw[split]["stories"]], tokenizer, context_length)
        tensor_path = out_dir / f"{split}_packed.pt"
        torch.save({"input_ids": inputs, "labels": labels}, tensor_path)
        result[split] = {
            **stats,
            "path": str(tensor_path),
            "sha256": sha256_file(tensor_path),
            "raw_path": q1_raw[split]["path"],
            "raw_sha256": q1_raw[split]["sha256"],
            "fixed_source": "ML-Q1 frozen split",
        }
    return result


def prepare_full_dataset() -> dict[str, Any]:
    frozen = _validate_frozen_inputs()
    tokenizer = load_tokenizer()
    cfg = HardwareMatchedConfig.from_json_dict(frozen["model_config"])
    data_dir = q2_root() / "data"
    packed_dir = data_dir / "packed"
    manifest_path = data_dir / "full_dataset_manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    data_dir.mkdir(parents=True, exist_ok=True)
    packed_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = data_dir / "raw" / "parquet"
    shard_records = [_ensure_shard(shard, shard_dir) for shard in official_train_shards()]

    counter = _SequenceCounter(tokenizer, cfg.context_length)
    seen: set[str] = set()
    duplicate_rows = 0
    empty_rows = 0
    excluded_holdout = 0
    total_rows = 0
    total_chars = 0
    long_rows: list[dict[str, Any]] = []
    shard_distribution = [{"filename": row["filename"], "used_train_rows": 0, "excluded_holdout_rows": 0} for row in shard_records]
    pass1_start = time.perf_counter()
    for global_index, shard_index, text in _iter_shard_texts(shard_records):
        total_rows += 1
        if not text:
            empty_rows += 1
            continue
        digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
        if digest in seen:
            duplicate_rows += 1
        else:
            seen.add(digest)
        if len(text) > 10000:
            long_rows.append({"global_index": global_index, "shard": shard_records[shard_index]["filename"], "chars": len(text)})
            long_rows = sorted(long_rows, key=lambda row: row["chars"], reverse=True)[:10]
        if HOLDOUT_GLOBAL_START <= global_index < HOLDOUT_GLOBAL_START + HOLDOUT_COUNT:
            excluded_holdout += 1
            shard_distribution[shard_index]["excluded_holdout_rows"] += 1
            continue
        total_chars += len(text)
        shard_distribution[shard_index]["used_train_rows"] += 1
        counter.add(text)
    counter.finalize()

    train_path = packed_dir / "train_packed.pt"
    pass2_start = time.perf_counter()
    existing_train = None
    if train_path.exists():
        existing_train = _torch_load(train_path)
        if tuple(existing_train["input_ids"].shape) != (counter.sequence_count, cfg.context_length):
            existing_train = None
    if existing_train is None:
        writer = _SequenceWriter(tokenizer, cfg.context_length, counter.sequence_count)
        for global_index, _shard_index, text in _iter_shard_texts(shard_records):
            if not text or HOLDOUT_GLOBAL_START <= global_index < HOLDOUT_GLOBAL_START + HOLDOUT_COUNT:
                continue
            writer.add(text)
        writer.finalize()
        torch.save({"input_ids": writer.inputs, "labels": writer.labels}, train_path)
        train_labels = writer.labels
        reused_existing_train = False
    else:
        train_labels = existing_train["labels"]
        reused_existing_train = True
    val_holdout = _pack_validation_holdout(tokenizer, cfg.context_length, packed_dir)
    non_ignored = int((train_labels != IGNORE_INDEX).sum().item())
    total_label_slots = int(train_labels.numel())
    manifest = {
        "stage": "ML-Q2",
        "dataset": {
            "name": "TinyStories",
            "source_url": TINYSTORIES_CARD_URL,
            "validation_url": TINYSTORIES_VALID_URL,
            "revision": HF_DATASET_REVISION,
            "license": TINYSTORIES_LICENSE,
            "access_date": "2026-07-13",
        },
        "official_train_shards": shard_records,
        "frozen_inputs": frozen,
        "tokenizer": {
            "tokenizer_json": load_interactive_bundle().data_manifest["tokenizer"]["tokenizer_json"],
            "tokenizer_sha256": sha256_file(load_interactive_bundle().data_manifest["tokenizer"]["tokenizer_json"]),
            "vocab_size": len(tokenizer.vocab),
            "special_tokens": {"PAD": tokenizer.pad_id, "BOS": tokenizer.bos_id, "EOS": tokenizer.eos_id, "UNK": tokenizer.unk_id},
        },
        "split_policy": {
            "train": "all official train parquet rows except empty rows and frozen ML-Q1 holdout global indices",
            "holdout_global_index_range": [HOLDOUT_GLOBAL_START, HOLDOUT_GLOBAL_START + HOLDOUT_COUNT - 1],
            "validation": "ML-Q1 frozen 10000-story validation split",
            "holdout": "ML-Q1 frozen 10000-story holdout split",
        },
        "quality_checks": {
            "official_train_rows": total_rows,
            "train_stories": total_rows - empty_rows - excluded_holdout,
            "excluded_holdout_rows": excluded_holdout,
            "empty_rows": empty_rows,
            "duplicate_rows": duplicate_rows,
            "duplicate_ratio": duplicate_rows / max(total_rows, 1),
            "abnormally_long_rows_top10": long_rows,
            "encoding_failures": 0,
            "total_train_characters": total_chars,
        },
        "packed": {
            "train": {
                "path": str(train_path),
                "sha256": sha256_file(train_path),
                "stories": total_rows - empty_rows - excluded_holdout,
                "total_encoded_tokens": counter.token_count,
                "packed_sequences": counter.sequence_count,
                "reused_existing_tensor": reused_existing_train,
                "context_length": cfg.context_length,
                "tokens_per_story": counter.token_count / max(total_rows - empty_rows - excluded_holdout, 1),
                "packing_utilization": non_ignored / max(total_label_slots, 1),
                "pad_ratio": 1.0 - non_ignored / max(total_label_slots, 1),
                "eos_count": counter.eos_count,
                "eos_ratio": counter.eos_count / max(counter.token_count, 1),
                "unk_count": counter.unk_count,
                "unk_ratio": counter.unk_count / max(counter.token_count, 1),
                "non_ignored_labels": non_ignored,
            },
            **val_holdout,
        },
        "shard_distribution": shard_distribution,
        "timing": {
            "count_pass_seconds": pass2_start - pass1_start,
            "pack_pass_seconds": time.perf_counter() - pass2_start,
        },
    }
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def _model_for_init(cfg: Q2ExperimentConfig) -> HardwareMatchedCausalLM:
    if cfg.init == "ml_q1_best":
        return _load_model_from_checkpoint(ML_Q1_CHECKPOINT)
    if cfg.init == "random":
        set_seed(cfg.seed)
        return HardwareMatchedCausalLM(HardwareMatchedConfig())
    raise ValueError(cfg.init)


def _save_q2_checkpoint(
    path: Path,
    model: HardwareMatchedCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    epoch: float,
    cfg: Q2ExperimentConfig,
    metrics: dict[str, Any],
    data_manifest: dict[str, Any],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "ML-Q2",
        "model_role": cfg.role,
        "model_name": MODEL_NAME,
        "step": step,
        "epoch": epoch,
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "config": model.config.to_json_dict(),
        "training_config": asdict(cfg),
        "metrics": metrics,
        "data_manifest": data_manifest,
        "frozen_inputs": data_manifest["frozen_inputs"],
    }
    torch.save(payload, path)
    manifest = {"path": str(path), "sha256": sha256_file(path), "step": step, "epoch": epoch, "metrics": metrics}
    _write_json(path.with_suffix(path.suffix + ".manifest.json"), manifest)
    return manifest


def _gpu_snapshot() -> dict[str, Any]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        util, mem_used, mem_total = [part.strip() for part in out.split(",")[:3]]
        return {"gpu_utilization_percent": float(util), "memory_used_mib": float(mem_used), "memory_total_mib": float(mem_total)}
    except Exception as exc:  # pragma: no cover - diagnostic only
        return {"error": str(exc)}


def _next_batch(loader: DataLoader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train_experiment(cfg: Q2ExperimentConfig) -> dict[str, Any]:
    data_manifest = prepare_full_dataset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("ML-Q2 requires CUDA for full-dataset training")
    use_bf16 = torch.cuda.is_bf16_supported()
    train_ds = _load_packed_dataset(data_manifest["packed"]["train"]["path"])
    val_ds = _load_packed_dataset(data_manifest["packed"]["validation"]["path"])
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
        pin_memory=True,
        num_workers=0,
    )
    steps_per_epoch = math.ceil(len(train_ds) / cfg.batch_size)
    max_steps = steps_per_epoch * cfg.max_epochs
    warmup_steps = max(1, int(max_steps * cfg.warmup_fraction))
    min_lr_ratio = cfg.minimum_lr / cfg.learning_rate
    exp_root = q2_root() / cfg.name
    training_dir = exp_root / "training"
    ckpt_dir = exp_root / "checkpoints"
    training_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def make_optimizer(model):
        return build_optimizer(
            model,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps,
            fused=True,
        )

    set_seed(cfg.seed)
    pilot_model = _model_for_init(cfg).to(device)
    pilot_optimizer = make_optimizer(pilot_model)
    pilot_scheduler = build_scheduler(
        pilot_optimizer,
        total_steps=max_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
        schedule="cosine",
    )
    pilot_iter = iter(train_loader)
    pilot_history = []
    for step in range(1, cfg.pilot_steps + 1):
        (inputs, labels), pilot_iter = _next_batch(train_loader, pilot_iter)
        inputs = inputs.long().to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)
        pilot_optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            loss = pilot_model(inputs, labels=labels)["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"{cfg.name} pilot loss is not finite")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(pilot_model.parameters(), cfg.grad_clip)
        pilot_optimizer.step()
        pilot_scheduler.step()
        if step in {1, 100, 250, cfg.pilot_steps}:
            pilot_history.append({"step": step, "loss": float(loss.item()), "grad_norm": float(grad_norm), "lr": float(pilot_scheduler.get_last_lr()[0])})
    pilot_val = evaluate_dataset_loss(pilot_model, val_ds, device=device)
    _save_q2_checkpoint(
        ckpt_dir / f"ml_q2_{cfg.name}_pilot.pt",
        pilot_model,
        pilot_optimizer,
        pilot_scheduler,
        cfg.pilot_steps,
        cfg.pilot_steps / steps_per_epoch,
        cfg,
        {"validation_loss": pilot_val["loss"], "pilot_history": pilot_history},
        data_manifest,
    )
    del pilot_model, pilot_optimizer, pilot_scheduler
    torch.cuda.empty_cache()

    set_seed(cfg.seed)
    model = _model_for_init(cfg).to(device)
    optimizer = make_optimizer(model)
    scheduler = build_scheduler(
        optimizer,
        total_steps=max_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
        schedule="cosine",
    )
    best_loss = float("inf")
    best_manifest = None
    history = []
    first100 = []
    step = 0
    train_iter = iter(train_loader)
    start = time.perf_counter()
    data_wait_seconds = 0.0
    last_epoch_losses: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    for epoch_index in range(1, cfg.max_epochs + 1):
        epoch_start = time.perf_counter()
        epoch_loss_sum = 0.0
        for _ in range(steps_per_epoch):
            wait_start = time.perf_counter()
            (inputs, labels), train_iter = _next_batch(train_loader, train_iter)
            data_wait_seconds += time.perf_counter() - wait_start
            step += 1
            inputs = inputs.long().to(device, non_blocking=True)
            labels = labels.long().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(inputs, labels=labels)["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"{cfg.name} formal loss is not finite at step {step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            loss_value = float(loss.item())
            epoch_loss_sum += loss_value
            if step <= 100 or step in {250, 500, 1000}:
                first100.append({"step": step, "loss": loss_value, "grad_norm": float(grad_norm), "lr": float(scheduler.get_last_lr()[0])})
        val = evaluate_dataset_loss(model, val_ds, device=device)
        epoch_loss = epoch_loss_sum / steps_per_epoch
        last_epoch_losses.append(val["loss"])
        elapsed = time.perf_counter() - start
        metrics = {
            "epoch": epoch_index,
            "step": step,
            "train_loss": epoch_loss,
            "validation_loss": val["loss"],
            "validation_perplexity": val["perplexity"],
            "lr": float(scheduler.get_last_lr()[0]),
            "elapsed_seconds": elapsed,
            "epoch_seconds": time.perf_counter() - epoch_start,
            "tokens_per_second": (step * cfg.batch_size * model.config.context_length) / max(elapsed, 1e-9),
            "sequences_per_second": (step * cfg.batch_size) / max(elapsed, 1e-9),
            "gpu": _gpu_snapshot(),
        }
        history.append(metrics)
        if val["loss"] < best_loss:
            best_loss = val["loss"]
            best_manifest = _save_q2_checkpoint(
                ckpt_dir / f"ml_q2_{cfg.name}_best.pt",
                model,
                optimizer,
                scheduler,
                step,
                float(epoch_index),
                cfg,
                metrics,
                data_manifest,
            )
        if epoch_index >= cfg.epochs:
            if epoch_index >= cfg.max_epochs:
                break
            previous = last_epoch_losses[-2] if len(last_epoch_losses) >= 2 else float("inf")
            improvement = (previous - last_epoch_losses[-1]) / max(previous, 1e-9)
            if improvement < cfg.optional_threshold:
                break
    elapsed = time.perf_counter() - start
    last_metrics = {
        "step": step,
        "epoch": history[-1]["epoch"],
        "elapsed_seconds": elapsed,
        "final_train_loss": history[-1]["train_loss"],
        "final_validation_loss": history[-1]["validation_loss"],
    }
    last_manifest = _save_q2_checkpoint(
        ckpt_dir / f"ml_q2_{cfg.name}_last.pt",
        model,
        optimizer,
        scheduler,
        step,
        float(history[-1]["epoch"]),
        cfg,
        last_metrics,
        data_manifest,
    )
    result = {
        "stage": "ML-Q2",
        "experiment": cfg.name,
        "status": "TRAINING_COMPLETE",
        "config": asdict(cfg),
        "steps_per_epoch": steps_per_epoch,
        "epochs_completed": history[-1]["epoch"],
        "total_steps": step,
        "train_packed_sequences": len(train_ds),
        "training_tokens": step * cfg.batch_size * model.config.context_length,
        "elapsed_seconds": elapsed,
        "tokens_per_second": (step * cfg.batch_size * model.config.context_length) / max(elapsed, 1e-9),
        "sequences_per_second": (step * cfg.batch_size) / max(elapsed, 1e-9),
        "data_wait_seconds": data_wait_seconds,
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved()),
        "pilot": {"history": pilot_history, "validation_loss": pilot_val["loss"]},
        "first100": first100,
        "history": history,
        "best_validation_loss": best_loss,
        "best_checkpoint": best_manifest["path"] if best_manifest else None,
        "best_sha256": best_manifest["sha256"] if best_manifest else None,
        "last_checkpoint": last_manifest["path"],
        "last_sha256": last_manifest["sha256"],
        "optimizer_reinitialized": True,
        "scheduler_restarted": True,
        "nan_or_inf": False,
    }
    _write_json(training_dir / f"{cfg.name}_training_metrics.json", result)
    return result


def train_all() -> dict[str, Any]:
    return {
        "continuation": train_experiment(CONTINUATION_CFG),
        "from_scratch": train_experiment(FROM_SCRATCH_CFG),
    }


def _bundle_from_checkpoint(name: str, checkpoint: str | Path) -> ModelBundle:
    baseline = load_interactive_bundle()
    payload = _torch_load(checkpoint)
    cfg = HardwareMatchedConfig.from_json_dict(payload["config"])
    model = HardwareMatchedCausalLM(cfg)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return ModelBundle(
        artifact_root=q2_root(),
        model=model,
        tokenizer=baseline.tokenizer,
        checkpoint_path=Path(checkpoint),
        checkpoint_sha256=sha256_file(checkpoint),
        data_manifest=baseline.data_manifest,
        training_metrics={"model_name": name},
        dataset_metadata=baseline.dataset_metadata,
        generation_config_path=baseline.generation_config_path,
        generation_config_manifest=baseline.generation_config_manifest,
    )


def _prompt_suite() -> list[str]:
    base = json.loads(Path("ml/evaluation/ml_m2_prompt_suite.json").read_text(encoding="utf-8"))["prompts"]
    extra = [
        "Anna wanted to make her mother smile",
        "The little car rolled down the hill",
        "Ben saw a shiny key under the bed",
        "Mia and her brother built a small house",
        "The rain made the garden smell fresh",
        "Sam could not find his blue hat",
        "A tiny frog jumped into the pond",
        "The teacher gave the class a surprise",
        "Nora opened the box very slowly",
        "The kitten heard a sound outside",
        "A warm light came from the window",
        "Jack promised to tell the truth",
        "The snowman began to melt in the sun",
        "Ella found a map behind the shelf",
        "The puppy was too small to climb",
        "Grandma baked a cake for everyone",
        "The moon looked bright over the farm",
        "Tom dropped his toy in the mud",
        "A soft voice called from the room",
        "The children followed the path home",
    ]
    holdout = split_tinystories_text(Path("D:/IC_Workspace/VEDA_artifacts/ml_q1/data/raw/TinyStories-holdout-after-300000-count-10000.txt").read_text(encoding="utf-8", errors="replace"))
    mids = []
    endings = []
    for story in holdout[:20]:
        words = story.split()
        if len(words) > 40:
            mids.append(" ".join(words[len(words) // 3 : len(words) // 3 + 20]))
            endings.append(" ".join(words[-28:-8]))
    return (base + extra + mids[:10] + endings[:10])[:50]


def _generation_variants() -> list[tuple[str, GenerationConfig]]:
    return [
        ("greedy", GenerationConfig(mode="greedy", temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0, max_new_tokens=128)),
        ("temperature_0_8", GenerationConfig(mode="sample", temperature=0.8, top_k=0, top_p=1.0, repetition_penalty=1.0, max_new_tokens=128)),
        ("temperature_1_0", GenerationConfig(mode="sample", temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0, max_new_tokens=128)),
        ("top_p_0_9", GenerationConfig(mode="sample", temperature=0.8, top_k=0, top_p=0.9, repetition_penalty=1.0, max_new_tokens=128)),
        ("top_p_0_95", GenerationConfig(mode="sample", temperature=0.8, top_k=0, top_p=0.95, repetition_penalty=1.0, max_new_tokens=128)),
        ("repetition_penalty_1_1", GenerationConfig(mode="sample", temperature=0.8, top_k=40, top_p=0.9, repetition_penalty=1.1, max_new_tokens=128)),
    ]


def _ngram_loop_count(tokens: list[int], n: int = 3) -> int:
    seen = set()
    loops = 0
    for idx in range(0, max(0, len(tokens) - n + 1)):
        gram = tuple(tokens[idx : idx + n])
        if gram in seen:
            loops += 1
        seen.add(gram)
    return loops


def _distinct(tokens: list[int], n: int) -> float:
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)


def evaluate_generation(name: str, bundle: ModelBundle, prompts: list[str]) -> dict[str, Any]:
    bundle.model.to("cpu")
    rows = []
    special = {bundle.tokenizer.pad_id, bundle.tokenizer.bos_id, bundle.tokenizer.eos_id, bundle.tokenizer.unk_id}
    for prompt in prompts:
        for variant, cfg in _generation_variants():
            record = generate_text_record(bundle, prompt, cfg)
            ids = record["generated_token_ids"]
            rows.append(
                {
                    **record,
                    "variant": variant,
                    "distinct_1": _distinct(ids, 1),
                    "distinct_2": _distinct(ids, 2),
                    "ngram_loop_count": _ngram_loop_count(ids),
                    "single_token_collapse": len(set(ids)) <= 2 and len(ids) > 8,
                    "special_token_ratio": sum(1 for token in ids if token in special) / max(len(ids), 1),
                }
            )
    summary = {}
    for variant, _ in _generation_variants():
        subset = [row for row in rows if row["variant"] == variant]
        summary[variant] = {
            "average_generated_length": sum(len(row["generated_token_ids"]) for row in subset) / max(len(subset), 1),
            "average_entropy": sum(row["average_entropy"] for row in subset) / max(len(subset), 1),
            "average_distinct_1": sum(row["distinct_1"] for row in subset) / max(len(subset), 1),
            "average_distinct_2": sum(row["distinct_2"] for row in subset) / max(len(subset), 1),
            "eos_rate": sum(1 for row in subset if row["hit_eos"]) / max(len(subset), 1),
            "average_special_token_ratio": sum(row["special_token_ratio"] for row in subset) / max(len(subset), 1),
            "single_token_collapse_count": sum(1 for row in subset if row["single_token_collapse"]),
            "ngram_loop_count": sum(row["ngram_loop_count"] for row in subset),
        }
    payload = {"model": name, "prompt_count": len(prompts), "variant_summary": summary, "results": rows}
    out = q2_root() / "evaluation" / f"{name}_generation.json"
    _write_json(out, payload)
    payload["output_path"] = str(out)
    return payload


@torch.no_grad()
def evaluate_eos(name: str, bundle: ModelBundle, max_examples: int = 1000) -> dict[str, Any]:
    docs = split_tinystories_text(Path("D:/IC_Workspace/VEDA_artifacts/datasets/tinystories_ml_m2/TinyStories-valid-prefix-10000.txt").read_text(encoding="utf-8", errors="replace"))[:max_examples]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = bundle.model.to(device)
    tokenizer = bundle.tokenizer
    ranks = []
    probs = []
    losses = []
    top1 = 0
    top5 = 0
    false_positive = 0
    nonterminal = 0
    for doc in docs:
        ids = tokenizer.encode(doc, add_bos=True, add_eos=True)
        prefix = ids[:-1][-model.config.context_length :]
        logits = model(torch.tensor([prefix], dtype=torch.long, device=device))["logits"][0, -1].float().cpu()
        p = torch.softmax(logits, dim=-1)
        sorted_ids = torch.argsort(p, descending=True)
        rank = int((sorted_ids == tokenizer.eos_id).nonzero(as_tuple=False)[0].item()) + 1
        prob = float(p[tokenizer.eos_id].item())
        ranks.append(rank)
        probs.append(prob)
        losses.append(-math.log(max(prob, 1e-30)))
        top1 += int(rank == 1)
        top5 += int(rank <= 5)
        if len(ids) > 16:
            mid = ids[: min(16, len(ids) - 1)]
            mid_logits = model(torch.tensor([mid], dtype=torch.long, device=device))["logits"][0, -1].float().cpu()
            false_positive += int(int(torch.argmax(mid_logits).item()) == tokenizer.eos_id)
            nonterminal += 1
    result = {
        "model": name,
        "examples": len(ranks),
        "eos_top1_accuracy": top1 / max(len(ranks), 1),
        "eos_top5_accuracy": top5 / max(len(ranks), 1),
        "eos_average_rank": sum(ranks) / max(len(ranks), 1),
        "eos_average_probability": sum(probs) / max(len(probs), 1),
        "eos_average_loss": sum(losses) / max(len(losses), 1),
        "non_terminal_eos_false_positive_rate": false_positive / max(nonterminal, 1),
    }
    _write_json(q2_root() / "evaluation" / f"{name}_eos.json", result)
    return result


def _ids_for_length(tokenizer: SimpleBPETokenizer, length: int) -> torch.Tensor:
    seed = "Once upon a time there was a small red bird who liked kind stories in the garden."
    ids = tokenizer.encode(seed, add_bos=True)
    filler = [idx for idx in ids if idx != tokenizer.eos_id]
    while len(filler) < length:
        filler.extend(filler[1:] or [tokenizer.bos_id])
    return torch.tensor([filler[:length]], dtype=torch.long)


def _checksum_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@torch.no_grad()
def hardware_aware_benchmark(bundle: ModelBundle, lengths: list[int] | None = None) -> dict[str, Any]:
    lengths = lengths or [1, 2, 8, 16, 32, 64]
    model = bundle.model.to("cpu")
    tokenizer = bundle.tokenizer
    fp16_model = _fp16_weight_rounded_model(model)
    results = {}
    for length in lengths:
        ids = _ids_for_length(tokenizer, length)
        pt = model(ids, return_trace=True)
        fp16 = fp16_model(ids)
        hw = run_hardware_aware_model(model, ids)
        inc = compare_full_vs_incremental(model, ids, atol=5.0e-5)
        hw_k = fp16_bits_nested_to_tensor(hw["k_cache"])
        hw_v = fp16_bits_nested_to_tensor(hw["v_cache"])
        results[f"len_{length}"] = {
            "pytorch_vs_fp16_weight": {**tensor_error_metrics(pt["logits"], fp16["logits"]), **logits_agreement(pt["logits"], fp16["logits"])},
            "pytorch_vs_hardware_aware": {**tensor_error_metrics(pt["logits"], hw["logits"]), **logits_agreement(pt["logits"], hw["logits"])},
            "incremental_full": inc,
            "incremental_acceptance_atol": 5.0e-5,
            "k_cache_checksum": _checksum_tensor(hw_k),
            "v_cache_checksum": _checksum_tensor(hw_v),
            "nan_or_inf": bool(torch.isnan(hw["logits"]).any().item() or torch.isinf(hw["logits"]).any().item()),
        }
    out = q2_root() / "benchmark" / "hardware_aware_comparison.json"
    _write_json(out, {"checkpoint": str(bundle.checkpoint_path), "checkpoint_sha256": bundle.checkpoint_sha256, "results": results})
    return {"path": str(out), "results": results}


def _load_training_metrics() -> dict[str, Any]:
    return {
        "continuation": json.loads((q2_root() / "continuation" / "training" / "continuation_training_metrics.json").read_text(encoding="utf-8")),
        "from_scratch": json.loads((q2_root() / "from_scratch" / "training" / "from_scratch_training_metrics.json").read_text(encoding="utf-8")),
    }


def compare_and_select() -> dict[str, Any]:
    data_manifest = prepare_full_dataset()
    metrics = _load_training_metrics()
    checkpoints = {
        "ml_m2_baseline": ML_M2_BASELINE_CHECKPOINT,
        "ml_q1_candidate": ML_Q1_CHECKPOINT,
        "q2_continuation": metrics["continuation"]["best_checkpoint"],
        "q2_from_scratch": metrics["from_scratch"]["best_checkpoint"],
    }
    bundles = {name: _bundle_from_checkpoint(name, path) for name, path in checkpoints.items()}
    val_ds = _load_packed_dataset(data_manifest["packed"]["validation"]["path"])
    holdout_ds = _load_packed_dataset(data_manifest["packed"]["holdout"]["path"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    losses = {"validation": {}, "holdout": {}}
    for name, bundle in bundles.items():
        losses["validation"][name] = evaluate_dataset_loss(bundle.model, val_ds, device=device)
    q2_names = ["q2_continuation", "q2_from_scratch"]
    provisional = min(q2_names, key=lambda name: losses["validation"][name]["loss"])
    for name, bundle in bundles.items():
        losses["holdout"][name] = evaluate_dataset_loss(bundle.model, holdout_ds, device=device)
    prompts = _prompt_suite()
    generation = {name: evaluate_generation(name, bundle, prompts) for name, bundle in bundles.items()}
    eos = {name: evaluate_eos(name, bundle) for name, bundle in bundles.items()}
    final_name = provisional
    final_bundle = bundles[final_name]
    hw = hardware_aware_benchmark(final_bundle)
    q1_val = losses["validation"]["ml_q1_candidate"]["loss"]
    final_val = losses["validation"][final_name]["loss"]
    q1_holdout = losses["holdout"]["ml_q1_candidate"]["loss"]
    final_holdout = losses["holdout"][final_name]["loss"]
    pass_status = (
        final_val < q1_val
        and final_holdout <= q1_holdout + 0.02
        and generation[final_name]["variant_summary"]["greedy"]["ngram_loop_count"]
        <= generation["ml_q1_candidate"]["variant_summary"]["greedy"]["ngram_loop_count"] * 1.1
        and eos[final_name]["eos_average_loss"] <= eos["ml_q1_candidate"]["eos_average_loss"]
        and eos[final_name]["eos_average_probability"] >= eos["ml_q1_candidate"]["eos_average_probability"]
        and eos[final_name]["eos_top5_accuracy"] >= eos["ml_q1_candidate"]["eos_top5_accuracy"] - 0.02
        and eos[final_name]["non_terminal_eos_false_positive_rate"]
        <= eos["ml_q1_candidate"]["non_terminal_eos_false_positive_rate"] + 0.01
        and all(row["pytorch_vs_hardware_aware"]["top1_agreement"] == 1.0 for row in hw["results"].values())
        and all(row["incremental_full"]["allclose"] for row in hw["results"].values())
    )
    result = {
        "stage": "ML-Q2",
        "selection_policy": "Q2 provisional best selected by validation loss only; holdout evaluated after provisional selection and was not used for tuning.",
        "checkpoints": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in checkpoints.items()},
        "losses": losses,
        "generation": {name: payload["variant_summary"] for name, payload in generation.items()},
        "eos": eos,
        "provisional_best": final_name,
        "quality_status": "ML-Q2 FULL-DATASET BENCHMARK PASS" if pass_status else "ML-Q2 NO SIGNIFICANT GAIN",
        "hardware_aware": hw,
    }
    _write_json(q2_root() / "evaluation" / "validation_holdout_generation_eos_comparison.json", result)
    return result


def export_benchmark() -> dict[str, Any]:
    comparison = json.loads((q2_root() / "evaluation" / "validation_holdout_generation_eos_comparison.json").read_text(encoding="utf-8"))
    final_name = comparison["provisional_best"]
    final_checkpoint = Path(comparison["checkpoints"][final_name]["path"])
    benchmark_dir = q2_root() / "benchmark"
    ckpt_dir = benchmark_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    benchmark_checkpoint = ckpt_dir / f"{MODEL_NAME}.pt"
    if not benchmark_checkpoint.exists() or sha256_file(benchmark_checkpoint) != sha256_file(final_checkpoint):
        shutil.copy2(final_checkpoint, benchmark_checkpoint)
    export_dir = benchmark_dir / "exports"
    export_manifest = export_checkpoint(benchmark_checkpoint, export_dir)
    validate_export_manifest(export_dir / "export_manifest.json")
    bundle = _bundle_from_checkpoint(MODEL_NAME, benchmark_checkpoint)
    trace_dir = benchmark_dir / "traces"
    trace_results = []
    for length in [1, 2, 8, 16, 32]:
        ids = _ids_for_length(bundle.tokenizer, length)
        manifest = export_trace(bundle.model, ids, trace_dir / f"trace_len_{length}.json")
        fixture = trace_dir / f"rtl_fixture_len_{length}.json"
        write_small_rtl_fixture(fixture, manifest)
        trace_results.append(
            {
                "prompt_length": length,
                "trace": str(trace_dir / f"trace_len_{length}.json"),
                "trace_sha256": sha256_file(trace_dir / f"trace_len_{length}.json"),
                "rtl_fixture": str(fixture),
                "rtl_fixture_sha256": sha256_file(fixture),
                "trace_node_count": manifest["trace_node_count"],
            }
        )
    result = {
        "model_name": MODEL_NAME,
        "source_checkpoint": str(final_checkpoint),
        "benchmark_checkpoint": str(benchmark_checkpoint),
        "benchmark_sha256": sha256_file(benchmark_checkpoint),
        "export_dir": str(export_dir),
        "export_manifest": str(export_dir / "export_manifest.json"),
        "export_manifest_sha256": sha256_file(export_dir / "export_manifest.json"),
        "export_tensor_count": export_manifest.tensor_count,
        "export_tensors": [record.logical_name for record in export_manifest.records],
        "trace_dir": str(trace_dir),
        "trace_results": trace_results,
    }
    _write_json(benchmark_dir / "benchmark_export_trace_summary.json", result)
    return result


def write_reports() -> None:
    root = q2_root()
    data = json.loads((root / "data" / "full_dataset_manifest.json").read_text(encoding="utf-8"))
    train = _load_training_metrics()
    comp = json.loads((root / "evaluation" / "validation_holdout_generation_eos_comparison.json").read_text(encoding="utf-8"))
    export = json.loads((root / "benchmark" / "benchmark_export_trace_summary.json").read_text(encoding="utf-8"))
    final = comp["provisional_best"]
    reports = Path("reports/ml_q2")
    reports.mkdir(parents=True, exist_ok=True)
    shard_lines = "\n".join(
        f"- {s['filename']}: rows={s.get('row_count')}, bytes={s['size_bytes']}, sha256={s['sha256']}"
        for s in data["official_train_shards"]
    )
    _write_md(
        reports / "full_dataset_manifest.md",
        f"""# ML-Q2 Full Dataset Manifest

Official TinyStories train shards:

{shard_lines}

```text
train_stories={data['packed']['train']['stories']}
official_train_rows={data['quality_checks']['official_train_rows']}
excluded_holdout_rows={data['quality_checks']['excluded_holdout_rows']}
duplicate_rows={data['quality_checks']['duplicate_rows']}
empty_rows={data['quality_checks']['empty_rows']}
total_encoded_tokens={data['packed']['train']['total_encoded_tokens']}
packed_sequences={data['packed']['train']['packed_sequences']}
packing_utilization={data['packed']['train']['packing_utilization']}
unk_ratio={data['packed']['train']['unk_ratio']}
```
""",
    )
    for name in ["continuation", "from_scratch"]:
        payload = train[name]
        _write_md(
            reports / f"{name}_training.md",
            f"""# ML-Q2 {name.replace('_', ' ').title()} Training

```text
init={payload['config']['init']}
epochs_completed={payload['epochs_completed']}
steps={payload['total_steps']}
training_tokens={payload['training_tokens']}
elapsed_seconds={payload['elapsed_seconds']}
tokens_per_second={payload['tokens_per_second']}
peak_allocated_vram_bytes={payload['peak_allocated_vram_bytes']}
best_validation_loss={payload['best_validation_loss']}
best_checkpoint={payload['best_checkpoint']}
best_sha256={payload['best_sha256']}
```
""",
        )
    val_lines = "\n".join(f"- {name}: loss={row['loss']}, ppl={row['perplexity']}" for name, row in comp["losses"]["validation"].items())
    holdout_lines = "\n".join(f"- {name}: loss={row['loss']}, ppl={row['perplexity']}" for name, row in comp["losses"]["holdout"].items())
    _write_md(reports / "validation_comparison.md", "# ML-Q2 Validation Comparison\n\n" + val_lines)
    _write_md(reports / "holdout_comparison.md", "# ML-Q2 Holdout Comparison\n\n" + holdout_lines + "\n\nHoldout was not used for checkpoint selection.")
    gen_lines = "\n".join(
        f"- {name}: greedy_collapse={row['greedy']['single_token_collapse_count']}, loops={row['greedy']['ngram_loop_count']}, distinct1={row['greedy']['average_distinct_1']}, distinct2={row['greedy']['average_distinct_2']}, eos_rate={row['greedy']['eos_rate']}"
        for name, row in comp["generation"].items()
    )
    _write_md(reports / "generation_comparison.md", "# ML-Q2 Generation Comparison\n\n" + gen_lines)
    eos_lines = "\n".join(
        f"- {name}: top1={row['eos_top1_accuracy']}, top5={row['eos_top5_accuracy']}, rank={row['eos_average_rank']}, prob={row['eos_average_probability']}"
        for name, row in comp["eos"].items()
    )
    _write_md(reports / "eos_comparison.md", "# ML-Q2 EOS Comparison\n\n" + eos_lines)
    hw_lines = "\n".join(
        f"- {length}: hw_top1={row['pytorch_vs_hardware_aware']['top1_agreement']}, hw_max_abs={row['pytorch_vs_hardware_aware']['max_abs_error']}, kv_allclose={row['incremental_full']['allclose']}"
        for length, row in comp["hardware_aware"]["results"].items()
    )
    _write_md(reports / "hardware_aware_comparison.md", "# ML-Q2 Hardware-Aware Comparison\n\n" + hw_lines)
    _write_md(
        reports / "benchmark_card.md",
        f"""# {MODEL_NAME} Benchmark Card

VEDA-HWLM-1L64-Q2 is an internal hardware benchmark model, not a general chat
model. It preserves the ML-M2/Q1 one-layer hardware-matched architecture:
decoder-only, RMSNorm, standard MHA, ReLU FFN, no bias, learned absolute
software-side position embedding, context 128, BPE-2048, tied embeddings.

```text
final_candidate={final}
checkpoint={export['benchmark_checkpoint']}
sha256={export['benchmark_sha256']}
train_stories={data['packed']['train']['stories']}
train_tokens={data['packed']['train']['total_encoded_tokens']}
validation_loss={comp['losses']['validation'][final]['loss']}
holdout_loss={comp['losses']['holdout'][final]['loss']}
export_dir={export['export_dir']}
trace_dir={export['trace_dir']}
```

Known limits: one transformer layer, `d_model=64`, context 128, modest generation
quality, and EOS generation is not guaranteed under greedy decoding. Intended
use is Model Stage M3 PyTorch / bit model / real RTL co-simulation.
""",
    )
    _write_md(
        reports / "artifact_manifest.md",
        f"""# ML-Q2 Artifact Manifest

```text
root={root}
data_manifest={root / 'data' / 'full_dataset_manifest.json'}
continuation_best={train['continuation']['best_checkpoint']}
from_scratch_best={train['from_scratch']['best_checkpoint']}
benchmark_checkpoint={export['benchmark_checkpoint']}
benchmark_sha256={export['benchmark_sha256']}
export_manifest={export['export_manifest']}
trace_dir={export['trace_dir']}
```
""",
    )
    _write_md(
        reports / "acceptance_audit.md",
        f"""# ML-Q2 Acceptance Audit

## Result

{comp['quality_status']}

```text
validation_q1={comp['losses']['validation']['ml_q1_candidate']['loss']}
validation_final={comp['losses']['validation'][final]['loss']}
holdout_q1={comp['losses']['holdout']['ml_q1_candidate']['loss']}
holdout_final={comp['losses']['holdout'][final]['loss']}
hardware_aware_top1_all={all(row['pytorch_vs_hardware_aware']['top1_agreement'] == 1.0 for row in comp['hardware_aware']['results'].values())}
incremental_kv_all={all(row['incremental_full']['allclose'] for row in comp['hardware_aware']['results'].values())}
export_tensor_count={export['export_tensor_count']}
```

ML-Q2 did not run real RTL and did not modify RTL or Hardware Stage H9 files.
""",
    )
    _write_md(
        reports / "summary.md",
        f"""# ML-Q2 Summary

```text
status={comp['quality_status']}
final={final}
benchmark={export['benchmark_checkpoint']}
sha256={export['benchmark_sha256']}
validation_loss={comp['losses']['validation'][final]['loss']}
holdout_loss={comp['losses']['holdout'][final]['loss']}
```

ML-Q2 remains a fixed-architecture quality experiment and does not start Model
Stage M3. The resulting benchmark is ready for later M3 co-simulation only if
the user approves that separate stage.
""",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare-data", "train-continuation", "train-from-scratch", "train-all", "compare", "export", "reports", "all"])
    args = parser.parse_args()
    if args.action == "prepare-data":
        print(json.dumps(prepare_full_dataset(), indent=2, sort_keys=True))
    elif args.action == "train-continuation":
        print(json.dumps(train_experiment(CONTINUATION_CFG), indent=2, sort_keys=True))
    elif args.action == "train-from-scratch":
        print(json.dumps(train_experiment(FROM_SCRATCH_CFG), indent=2, sort_keys=True))
    elif args.action == "train-all":
        print(json.dumps(train_all(), indent=2, sort_keys=True))
    elif args.action == "compare":
        print(json.dumps(compare_and_select(), indent=2, sort_keys=True))
    elif args.action == "export":
        print(json.dumps(export_benchmark(), indent=2, sort_keys=True))
    elif args.action == "reports":
        write_reports()
    elif args.action == "all":
        prepare_full_dataset()
        train_all()
        compare_and_select()
        export_benchmark()
        write_reports()


if __name__ == "__main__":
    main()
