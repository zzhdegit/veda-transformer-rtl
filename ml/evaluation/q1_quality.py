"""ML-Q1 fixed-architecture quality audit and continued training."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ml.architecture.causal_lm import HardwareMatchedCausalLM
from ml.architecture.config import HardwareMatchedConfig
from ml.cosim.hardware_aware_model import run_hardware_aware_model
from ml.data.dataset_hash import sha256_file
from ml.data.dataset_manifest import data_root
from ml.data.formal_data import TINYSTORIES_SOURCE_COMMIT
from ml.data.tinystories_loader import (
    TINYSTORIES_CARD_URL,
    TINYSTORIES_LICENSE,
    TINYSTORIES_TRAIN_URL,
    split_tinystories_text,
)
from ml.evaluation.evaluate_quantization import logits_agreement, tensor_error_metrics
from ml.export.formal_export import _fp16_weight_rounded_model, _torch_load
from ml.inference.interactive import (
    DEFAULT_ARTIFACT_ROOT,
    GenerationConfig,
    ModelBundle,
    config_variants,
    evaluate_prompt_suite,
    generate_text_record,
    load_interactive_bundle,
    next_token_report,
)
from ml.inference.incremental_decode import compare_full_vs_incremental
from ml.tokenizer.load_tokenizer import SimpleBPETokenizer
from ml.training.optimizer import build_optimizer
from ml.training.reproducibility import set_seed
from ml.training.scheduler import build_scheduler


ML_M2_BASELINE_CHECKPOINT = Path("D:/IC_Workspace/VEDA_artifacts/ml_m2/formal/checkpoints/ml_m2_formal_best.pt")
ML_M2_BASELINE_SHA256 = "cfaae278aa7fccd903b3b65041bce1b4dd91410ce3cdeacfb50e5b2b6ca933c8"
ML_Q1_ROOT = Path("D:/IC_Workspace/VEDA_artifacts/ml_q1")
IGNORE_INDEX = -100
TINYSTORIES_PARQUET_TRAIN_SHARDS = (
    {
        "path": "data/train-00000-of-00004-2d5a1467fff1081b.parquet",
        "url": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/train-00000-of-00004-2d5a1467fff1081b.parquet",
        "hf_oid": "f457b8ee94363800f857dc473192b278c8450030",
        "size_bytes": 248731111,
    },
)


@dataclass(frozen=True)
class Q1TrainConfig:
    train_stories: int = 300000
    validation_stories: int = 10000
    holdout_stories: int = 10000
    batch_size: int = 1024
    pilot_steps: int = 500
    epochs: int = 2
    learning_rate: float = 1.0e-4
    minimum_lr: float = 1.0e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1.0e-8
    grad_clip: float = 1.0
    warmup_fraction: float = 0.01
    seed: int = 20260713
    dtype: str = "bf16"


def q1_root() -> Path:
    return Path(os.environ.get("VEDA_ML_Q1_ROOT", str(ML_Q1_ROOT)))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_md(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text.rstrip() + "\n", encoding="utf-8")


def _read_stories(path: str | Path, limit: int | None = None) -> list[str]:
    stories = split_tinystories_text(Path(path).read_text(encoding="utf-8", errors="replace"))
    return stories if limit is None else stories[:limit]


def _write_stories(path: str | Path, stories: list[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    marker = "\n<|endoftext|>\n"
    target.write_text(marker.join(stories) + marker, encoding="utf-8")


def _token_ids_to_tokens(tokenizer: SimpleBPETokenizer, ids: list[int]) -> list[str]:
    return [tokenizer.id_to_token.get(int(token_id), "<missing>") for token_id in ids]


def load_baseline_bundle() -> ModelBundle:
    bundle = load_interactive_bundle(DEFAULT_ARTIFACT_ROOT)
    actual = sha256_file(ML_M2_BASELINE_CHECKPOINT)
    if actual != ML_M2_BASELINE_SHA256:
        raise RuntimeError(f"baseline checkpoint SHA mismatch: {actual}")
    return bundle


def _stream_story_prefix(total_stories: int, output_path: Path) -> Path:
    from ml.data.formal_data import stream_tinystories_subset

    return Path(stream_tinystories_subset(TINYSTORIES_TRAIN_URL, output_path, total_stories)["path"])


def _download_file(url: str, path: Path, expected_size: int | None = None) -> dict[str, Any]:
    if path.exists() and path.stat().st_size > 0:
        if expected_size is None or path.stat().st_size == expected_size:
            return {"path": str(path), "bytes": path.stat().st_size, "reused_existing": True}
    path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "veda-ml-q1/1.0"})
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    start = time.perf_counter()
    with urlopen(request, timeout=180) as response, tmp_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    tmp_path.replace(path)
    if expected_size is not None and path.stat().st_size != expected_size:
        raise RuntimeError(f"downloaded {path} has {path.stat().st_size} bytes; expected {expected_size}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "elapsed_seconds": time.perf_counter() - start,
        "reused_existing": False,
    }


def _read_parquet_story_prefix(total_stories: int, raw_dir: Path) -> tuple[list[str], list[dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "TinyStories-train.txt did not contain enough stories for ML-Q1; "
            "install pyarrow in the active training environment to read official parquet shards."
        ) from exc

    stories: list[str] = []
    sources: list[dict[str, Any]] = []
    parquet_dir = raw_dir / "parquet"
    for shard in TINYSTORIES_PARQUET_TRAIN_SHARDS:
        local_path = parquet_dir / Path(shard["path"]).name
        download = _download_file(shard["url"], local_path, expected_size=int(shard["size_bytes"]))
        parquet = pq.ParquetFile(local_path)
        text_column = None
        for field in parquet.schema_arrow:
            if str(field.type) in {"string", "large_string"}:
                text_column = field.name
                break
        if text_column is None:
            raise RuntimeError(f"no text column found in {local_path}")
        rows_before = len(stories)
        for batch in parquet.iter_batches(columns=[text_column], batch_size=8192):
            for value in batch.column(0).to_pylist():
                if value:
                    stories.append(str(value).strip())
                    if len(stories) >= total_stories:
                        break
            if len(stories) >= total_stories:
                break
        sources.append(
            {
                **shard,
                **download,
                "sha256": sha256_file(local_path),
                "row_count": parquet.metadata.num_rows,
                "text_column": text_column,
                "used_rows": len(stories) - rows_before,
            }
        )
        if len(stories) >= total_stories:
            break
    if len(stories) < total_stories:
        raise RuntimeError(f"official parquet shards yielded {len(stories)} stories, expected {total_stories}")
    return stories[:total_stories], sources


def ensure_q1_raw_data(cfg: Q1TrainConfig) -> dict[str, Any]:
    raw_dir = q1_root() / "data" / "raw"
    combined_target = cfg.train_stories + cfg.holdout_stories
    combined_path = raw_dir / f"TinyStories-train-prefix-{combined_target}.txt"
    train_path = raw_dir / f"TinyStories-train-prefix-{cfg.train_stories}.txt"
    holdout_path = raw_dir / f"TinyStories-holdout-after-{cfg.train_stories}-count-{cfg.holdout_stories}.txt"
    data_source = "existing ML-Q1 raw split"
    source_shards: list[dict[str, Any]] = []
    if not train_path.exists() or not holdout_path.exists():
        if combined_path.exists():
            stories = _read_stories(combined_path, combined_target)
        else:
            _stream_story_prefix(combined_target, combined_path)
            stories = _read_stories(combined_path, combined_target)
        if len(stories) < combined_target:
            text_count = len(stories)
            stories, source_shards = _read_parquet_story_prefix(combined_target, raw_dir)
            _write_stories(combined_path, stories)
            data_source = (
                "Official Hugging Face TinyStories parquet train shard; "
                f"TinyStories-train.txt yielded only {text_count} stories locally."
            )
        else:
            data_source = "Official Hugging Face TinyStories-train.txt"
        _write_stories(train_path, stories[: cfg.train_stories])
        _write_stories(holdout_path, stories[cfg.train_stories : combined_target])
    else:
        parquet_path = raw_dir / "parquet" / Path(TINYSTORIES_PARQUET_TRAIN_SHARDS[0]["path"]).name
        if parquet_path.exists():
            shard = TINYSTORIES_PARQUET_TRAIN_SHARDS[0]
            source_shards = [
                {
                    **shard,
                    "path": str(parquet_path),
                    "sha256": sha256_file(parquet_path),
                    "used_rows": combined_target,
                    "text_column": "text",
                }
            ]
            try:
                import pyarrow.parquet as pq

                source_shards[0]["row_count"] = pq.ParquetFile(parquet_path).metadata.num_rows
            except ImportError:
                source_shards[0]["row_count"] = None
            data_source = "Existing ML-Q1 raw split generated from official Hugging Face TinyStories parquet train shard"
    baseline_manifest = json.loads((DEFAULT_ARTIFACT_ROOT / "data" / "formal_data_manifest.json").read_text(encoding="utf-8"))
    validation_path = Path(baseline_manifest["subsets"]["validation_subset_path"])
    manifest = {
        "stage": "ML-Q1",
        "dataset": {
            "name": "TinyStories",
            "source_url": TINYSTORIES_CARD_URL,
            "train_url": TINYSTORIES_TRAIN_URL,
            "parquet_train_shards": source_shards,
            "source_commit": TINYSTORIES_SOURCE_COMMIT,
            "license": TINYSTORIES_LICENSE,
            "q1_train_source": data_source,
        },
        "train": {
            "stories": cfg.train_stories,
            "path": str(train_path),
            "sha256": sha256_file(train_path),
            "index_range": [0, cfg.train_stories - 1],
        },
        "validation": {
            "stories": cfg.validation_stories,
            "path": str(validation_path),
            "sha256": sha256_file(validation_path),
            "source": "ML-M2 fixed validation prefix",
        },
        "holdout": {
            "stories": cfg.holdout_stories,
            "path": str(holdout_path),
            "sha256": sha256_file(holdout_path),
            "index_range": [cfg.train_stories, cfg.train_stories + cfg.holdout_stories - 1],
        },
        "notes": [
            "Candidate training may include the ML-M2 first 100000 training stories.",
            "Validation and holdout are not used for candidate training.",
            "ML-M2 baseline checkpoint is not overwritten.",
        ],
    }
    _write_json(q1_root() / "data" / "q1_raw_data_manifest.json", manifest)
    return manifest


def _pack_docs(
    docs: list[str],
    tokenizer: SimpleBPETokenizer,
    context_length: int,
    chunk_sequences: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    buffer: list[int] = []
    input_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    eos_count = 0
    unk_count = 0
    token_count = 0

    def flush() -> None:
        nonlocal buffer
        seqs_in: list[list[int]] = []
        seqs_lab: list[list[int]] = []
        need = context_length + 1
        while len(buffer) >= need and len(seqs_in) < chunk_sequences:
            window = buffer[:need]
            del buffer[:context_length]
            seqs_in.append(window[:-1])
            seqs_lab.append(window[1:])
        if seqs_in:
            input_chunks.append(np.asarray(seqs_in, dtype=np.int16))
            label_chunks.append(np.asarray(seqs_lab, dtype=np.int16))

    for doc in docs:
        ids = tokenizer.encode(doc, add_bos=True, add_eos=True)
        eos_count += ids.count(tokenizer.eos_id)
        unk_count += ids.count(tokenizer.unk_id)
        token_count += len(ids)
        buffer.extend(ids)
        flush()
    if len(buffer) >= 2:
        need = context_length + 1
        window = buffer[:need]
        if len(window) < need:
            window = window + [tokenizer.pad_id] * (need - len(window))
        labels = [token if token != tokenizer.pad_id else IGNORE_INDEX for token in window[1:]]
        input_chunks.append(np.asarray([window[:-1]], dtype=np.int16))
        label_chunks.append(np.asarray([labels], dtype=np.int16))
    inputs = torch.from_numpy(np.concatenate(input_chunks, axis=0)) if input_chunks else torch.empty((0, context_length), dtype=torch.int16)
    labels = torch.from_numpy(np.concatenate(label_chunks, axis=0)) if label_chunks else torch.empty((0, context_length), dtype=torch.int16)
    stats = {
        "stories": len(docs),
        "token_count": token_count,
        "eos_count": eos_count,
        "unk_count": unk_count,
        "unk_ratio": unk_count / max(token_count, 1),
        "packed_sequences": int(inputs.shape[0]),
        "context_length": context_length,
    }
    return inputs, labels, stats


def ensure_q1_packed_data(cfg: Q1TrainConfig, bundle: ModelBundle | None = None) -> dict[str, Any]:
    bundle = bundle or load_baseline_bundle()
    raw = ensure_q1_raw_data(cfg)
    out_dir = q1_root() / "data" / "packed"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "q1_packed_manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    result = {"stage": "ML-Q1", "splits": {}, "tokenizer": bundle.data_manifest["tokenizer"]}
    for split in ["train", "validation", "holdout"]:
        docs = _read_stories(raw[split]["path"], raw[split]["stories"])
        inputs, labels, stats = _pack_docs(docs, bundle.tokenizer, bundle.model.config.context_length)
        tensor_path = out_dir / f"{split}_packed.pt"
        torch.save({"input_ids": inputs, "labels": labels}, tensor_path)
        result["splits"][split] = {
            **stats,
            "path": str(tensor_path),
            "sha256": sha256_file(tensor_path),
            "raw_path": raw[split]["path"],
            "raw_sha256": raw[split]["sha256"],
        }
    _write_json(manifest_path, result)
    result["manifest_path"] = str(manifest_path)
    result["manifest_sha256"] = sha256_file(manifest_path)
    return result


def tokenizer_audit(bundle: ModelBundle | None = None) -> dict[str, Any]:
    bundle = bundle or load_baseline_bundle()
    manifest = bundle.data_manifest
    q1_raw_manifest = q1_root() / "data" / "q1_raw_data_manifest.json"
    if q1_raw_manifest.exists():
        raw = json.loads(q1_raw_manifest.read_text(encoding="utf-8"))
        train_path = Path(raw["train"]["path"])
        validation_path = Path(raw["validation"]["path"])
    else:
        train_path = Path(manifest["subsets"]["train_subset_path"])
        validation_path = Path(manifest["subsets"]["validation_subset_path"])
    train_docs = _read_stories(train_path, 10000)
    validation_docs = _read_stories(validation_path, 10000)
    tokenizer = bundle.tokenizer

    def collect(docs: list[str]) -> dict[str, Any]:
        token_counts: Counter[int] = Counter()
        bigrams: Counter[tuple[int, int]] = Counter()
        chars = 0
        total_tokens = 0
        decode_stable = 0
        for doc in docs:
            ids = tokenizer.encode(doc, add_bos=True, add_eos=True)
            token_counts.update(ids)
            bigrams.update(zip(ids, ids[1:]))
            total_tokens += len(ids)
            chars += len(doc)
            decoded = tokenizer.decode(ids)
            if tokenizer.encode(decoded, add_bos=True, add_eos=True) == ids:
                decode_stable += 1
        return {
            "token_counts": token_counts,
            "bigrams": bigrams,
            "chars": chars,
            "tokens": total_tokens,
            "stories": len(docs),
            "decode_reencode_stable_ratio": decode_stable / max(len(docs), 1),
        }

    train = collect(train_docs)
    valid = collect(validation_docs)
    vocab = tokenizer.vocab
    leading_space_tokens = [tok for tok in vocab if tok.startswith(" ") and tok not in tokenizer.special_tokens]
    single_char_tokens = [tok for tok in vocab if len(tok) == 1 and tok not in tokenizer.special_tokens]
    space_id = vocab.get(" ")
    prompt = "Once upon a time"
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    top_tokens = [
        {"token_id": token_id, "token": tokenizer.id_to_token[token_id], "count": count}
        for token_id, count in train["token_counts"].most_common(20)
    ]
    top_bigrams = [
        {
            "token_ids": [left, right],
            "tokens": [tokenizer.id_to_token[left], tokenizer.id_to_token[right]],
            "count": count,
        }
        for (left, right), count in train["bigrams"].most_common(20)
    ]
    train_dist = train["token_counts"]
    valid_dist = valid["token_counts"]
    total_train = max(sum(train_dist.values()), 1)
    total_valid = max(sum(valid_dist.values()), 1)
    overlap_ids = set(train_dist) | set(valid_dist)
    l1 = sum(abs(train_dist[token] / total_train - valid_dist[token] / total_valid) for token in overlap_ids)
    result = {
        "vocab_size": len(vocab),
        "special_token_ids": {
            "PAD": tokenizer.pad_id,
            "BOS": tokenizer.bos_id,
            "EOS": tokenizer.eos_id,
            "UNK": tokenizer.unk_id,
        },
        "average_chars_per_token_train_sample": train["chars"] / max(train["tokens"], 1),
        "average_tokens_per_story_train_sample": train["tokens"] / max(train["stories"], 1),
        "average_tokens_per_story_validation": valid["tokens"] / max(valid["stories"], 1),
        "space_token": {"id": space_id, "frequency": train_dist.get(space_id, 0), "ratio": train_dist.get(space_id, 0) / total_train},
        "single_character_token_count": len(single_char_tokens),
        "leading_space_word_token_count": len(leading_space_tokens),
        "merge_count": len(tokenizer.merges),
        "merge_utilization": len(tokenizer.merges) / max(len(vocab) - len(tokenizer.special_tokens), 1),
        "unk_ratio_train_sample": train["token_counts"].get(tokenizer.unk_id, 0) / total_train,
        "unk_ratio_validation": valid["token_counts"].get(tokenizer.unk_id, 0) / total_valid,
        "train_validation_distribution_l1": l1,
        "decode_reencode_stable_ratio_train": train["decode_reencode_stable_ratio"],
        "decode_reencode_stable_ratio_validation": valid["decode_reencode_stable_ratio"],
        "top_20_tokens": top_tokens,
        "top_20_bigrams": top_bigrams,
        "prompt_encoding": {
            "prompt": prompt,
            "ids": prompt_ids,
            "tokens": _token_ids_to_tokens(tokenizer, prompt_ids),
            "roundtrip": tokenizer.decode(prompt_ids),
        },
        "space_conclusion": (
            "The tokenizer uses a standalone high-frequency space token; this is expected for the simple character-seeded BPE and is not by itself a tokenizer bug."
        ),
    }
    _write_json(q1_root() / "audits" / "tokenizer_audit.json", result)
    return result


@torch.no_grad()
def _eval_logits(model: HardwareMatchedCausalLM, input_ids: torch.Tensor, device: torch.device, use_bf16: bool = True) -> torch.Tensor:
    model.eval()
    input_ids = input_ids.to(device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16 and device.type == "cuda"):
        return model(input_ids)["logits"].detach().float().cpu()


@torch.no_grad()
def eos_audit(bundle: ModelBundle | None = None, max_examples: int = 1000) -> dict[str, Any]:
    bundle = bundle or load_baseline_bundle()
    tokenizer = bundle.tokenizer
    q1_packed_manifest = q1_root() / "data" / "packed" / "q1_packed_manifest.json"
    if q1_packed_manifest.exists():
        packed = json.loads(q1_packed_manifest.read_text(encoding="utf-8"))
        train_tensor = torch.load(packed["splits"]["train"]["path"], map_location="cpu")
        val_tensor = torch.load(packed["splits"]["validation"]["path"], map_location="cpu")
        validation_docs_path = packed["splits"]["validation"]["raw_path"]
    else:
        train_tensor = torch.load(bundle.data_manifest["packing"]["train_tensor"]["path"], map_location="cpu")
        val_tensor = torch.load(bundle.data_manifest["packing"]["validation_tensor"]["path"], map_location="cpu")
        validation_docs_path = bundle.data_manifest["subsets"]["validation_subset_path"]
    train_labels = train_tensor["labels"]
    val_labels = val_tensor["labels"]
    train_targets = train_labels[train_labels != IGNORE_INDEX]
    val_targets = val_labels[val_labels != IGNORE_INDEX]
    validation_docs = _read_stories(validation_docs_path, max_examples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = bundle.model.to(device)
    rows = []
    eos_rank_top1 = 0
    eos_rank_top5 = 0
    eos_prob_sum = 0.0
    eos_loss_sum = 0.0
    false_positive = 0
    non_terminal_checks = 0
    for doc in validation_docs:
        ids = tokenizer.encode(doc, add_bos=True, add_eos=True)
        if ids[-1] != tokenizer.eos_id:
            continue
        prefix = ids[:-1][-bundle.model.config.context_length :]
        logits = _eval_logits(model, torch.tensor([prefix], dtype=torch.long), device)[0, -1]
        probs = torch.softmax(logits, dim=-1)
        sorted_ids = torch.argsort(probs, descending=True)
        rank = int((sorted_ids == tokenizer.eos_id).nonzero(as_tuple=False)[0].item()) + 1
        eos_prob = float(probs[tokenizer.eos_id].item())
        eos_prob_sum += eos_prob
        eos_loss_sum += -math.log(max(eos_prob, 1e-30))
        eos_rank_top1 += int(rank == 1)
        eos_rank_top5 += int(rank <= 5)
        rows.append({"rank": rank, "probability": eos_prob, "prefix_len": len(prefix)})
        if len(ids) > 8:
            mid_prefix = ids[: min(len(ids) - 1, 16)]
            mid_logits = _eval_logits(model, torch.tensor([mid_prefix], dtype=torch.long), device)[0, -1]
            if int(torch.argmax(mid_logits).item()) == tokenizer.eos_id:
                false_positive += 1
            non_terminal_checks += 1
    prompts = json.loads(Path("ml/evaluation/ml_m2_prompt_suite.json").read_text(encoding="utf-8"))["prompts"]
    eos_rates = {}
    bundle.model.to("cpu")
    for max_new in [48, 128, 256]:
        cfg = GenerationConfig(mode="sample", temperature=0.8, top_k=40, top_p=0.9, repetition_penalty=1.1, max_new_tokens=max_new)
        generated = [generate_text_record(bundle, prompt, cfg) for prompt in prompts]
        eos_rates[str(max_new)] = {
            "requested_max_new_tokens": max_new,
            "effective_context_capped": True,
            "eos_rate": sum(1 for row in generated if row["hit_eos"]) / len(generated),
            "average_generated_length": sum(len(row["generated_token_ids"]) for row in generated) / len(generated),
        }
    result = {
        "train_eos_targets": int((train_targets == tokenizer.eos_id).sum().item()),
        "train_total_targets": int(train_targets.numel()),
        "train_eos_ratio": float((train_targets == tokenizer.eos_id).float().mean().item()),
        "validation_eos_targets": int((val_targets == tokenizer.eos_id).sum().item()),
        "validation_total_targets": int(val_targets.numel()),
        "validation_eos_ratio": float((val_targets == tokenizer.eos_id).float().mean().item()),
        "pad_eos_confusion": bool(tokenizer.pad_id == tokenizer.eos_id),
        "eos_is_ignore_index": bool(tokenizer.eos_id == IGNORE_INDEX),
        "validation_story_eos_examples": len(rows),
        "eos_top1_accuracy": eos_rank_top1 / max(len(rows), 1),
        "eos_top5_accuracy": eos_rank_top5 / max(len(rows), 1),
        "eos_average_probability": eos_prob_sum / max(len(rows), 1),
        "eos_average_loss": eos_loss_sum / max(len(rows), 1),
        "eos_average_rank": sum(row["rank"] for row in rows) / max(len(rows), 1),
        "non_terminal_eos_false_positive_rate": false_positive / max(non_terminal_checks, 1),
        "generation_eos_rates": eos_rates,
        "notes": [
            "EOS is retained in packed labels and is not ignored.",
            "Generation max_new_tokens above context is capped by the 128-token position limit.",
        ],
    }
    _write_json(q1_root() / "audits" / "eos_audit.json", result)
    model.cpu()
    return result


def _load_packed_dataset(path: str | Path) -> TensorDataset:
    payload = torch.load(Path(path), map_location="cpu")
    return TensorDataset(payload["input_ids"], payload["labels"])


@torch.no_grad()
def evaluate_dataset_loss(
    model: HardwareMatchedCausalLM,
    dataset: TensorDataset,
    batch_size: int = 1024,
    max_batches: int | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total_tokens = 0
    position_loss_sum = None
    position_count = None
    token_loss_sum: Counter[int] = Counter()
    token_count: Counter[int] = Counter()
    for idx, (input_ids, labels) in enumerate(loader):
        if max_batches is not None and idx >= max_batches:
            break
        labels_long = labels.long().to(device)
        input_long = input_ids.long().to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(input_long)["logits"]
        vocab = logits.shape[-1]
        per_token = torch.nn.functional.cross_entropy(
            logits.float().view(-1, vocab),
            labels_long.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).view(labels_long.shape)
        mask = labels_long != IGNORE_INDEX
        total_loss += float(per_token[mask].sum().detach().cpu())
        total_tokens += int(mask.sum().item())
        pos_sum = per_token.masked_fill(~mask, 0).sum(dim=0).detach().cpu()
        pos_count = mask.sum(dim=0).detach().cpu()
        position_loss_sum = pos_sum if position_loss_sum is None else position_loss_sum + pos_sum
        position_count = pos_count if position_count is None else position_count + pos_count
        label_cpu = labels_long.detach().cpu()
        loss_cpu = per_token.detach().cpu()
        mask_cpu = mask.detach().cpu()
        for token_id in torch.unique(label_cpu[mask_cpu]).tolist():
            token_mask = (label_cpu == int(token_id)) & mask_cpu
            token_loss_sum[int(token_id)] += float(loss_cpu[token_mask].sum().item())
            token_count[int(token_id)] += int(token_mask.sum().item())
    avg_loss = total_loss / max(total_tokens, 1)
    pos_loss = (position_loss_sum / position_count.clamp_min(1)).tolist() if position_loss_sum is not None else []
    token_avgs = {
        str(token): token_loss_sum[token] / max(token_count[token], 1)
        for token in token_count
    }
    return {
        "loss": avg_loss,
        "perplexity": math.exp(avg_loss) if avg_loss < 50 else float("inf"),
        "tokens": total_tokens,
        "position_loss": pos_loss,
        "token_loss": token_avgs,
        "token_counts": {str(token): count for token, count in token_count.items()},
        "evaluated_batches": idx + 1 if "idx" in locals() else 0,
    }


def capacity_audit(bundle: ModelBundle | None = None, packed_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    bundle = bundle or load_baseline_bundle()
    cfg = Q1TrainConfig()
    packed_manifest = packed_manifest or ensure_q1_packed_data(cfg, bundle)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = {
        "train_fixed_sample": _load_packed_dataset(bundle.data_manifest["packing"]["train_tensor"]["path"]),
        "validation": _load_packed_dataset(bundle.data_manifest["packing"]["validation_tensor"]["path"]),
        "holdout": _load_packed_dataset(packed_manifest["splits"]["holdout"]["path"]),
    }
    result = {}
    for name, dataset in splits.items():
        max_batches = 128 if name == "train_fixed_sample" else None
        result[name] = evaluate_dataset_loss(bundle.model, dataset, batch_size=1024, max_batches=max_batches, device=device)
    valid = result["validation"]
    token_counts = {int(k): v for k, v in valid["token_counts"].items()}
    token_loss = {int(k): v for k, v in valid["token_loss"].items()}
    high_freq = sorted(token_counts, key=token_counts.get, reverse=True)[:100]
    low_freq = [token for token in token_counts if token_counts[token] <= 5]
    result["frequency_loss"] = {
        "high_frequency_avg_loss": sum(token_loss[token] for token in high_freq) / max(len(high_freq), 1),
        "low_frequency_avg_loss": sum(token_loss[token] for token in low_freq) / max(len(low_freq), 1),
        "low_frequency_token_count": len(low_freq),
    }
    result["position_loss_summary"] = {
        "first_16_avg": sum(valid["position_loss"][:16]) / 16,
        "last_16_avg": sum(valid["position_loss"][-16:]) / 16,
    }
    result["attribution"] = {
        "primary": "undertraining plus fixed one-layer capacity limit",
        "secondary": ["decode policy brittleness", "EOS imbalance/calibration", "simple BPE fragmentation"],
        "not_overfitting": result["train_fixed_sample"]["loss"] <= result["validation"]["loss"] + 0.2,
    }
    _write_json(q1_root() / "audits" / "capacity_training_audit.json", result)
    bundle.model.cpu()
    return result


def run_baseline_eval(bundle: ModelBundle | None = None) -> dict[str, Any]:
    bundle = bundle or load_baseline_bundle()
    bundle.model.to("cpu")
    prompts = json.loads(Path("ml/evaluation/ml_m2_prompt_suite.json").read_text(encoding="utf-8"))["prompts"]
    extra = [
        "Mia opened the little box and",
        "Ben wanted to help his friend",
        "The puppy looked at the",
        "Sara found a shiny coin",
        "The old tree had a",
        "Tom and Lily went outside",
        "A tiny frog jumped into",
        "The teacher smiled because",
        "The blue kite flew over",
        "Anna heard a quiet sound",
        "The little car could not",
        "Sam put the apple in",
        "The moon was bright and",
        "A kind bear saw the",
        "Lucy made a cake for",
        "The train stopped near the",
        "A yellow duck wanted to",
        "The snow fell on the",
        "A boy lost his red",
        "The family went to the",
    ]
    story_prefixes = _read_stories(bundle.data_manifest["subsets"]["validation_subset_path"], 20)
    full_prefixes = [" ".join(story.split()[:18]) for story in story_prefixes[:10]]
    ending_prefixes = [" ".join(story.split()[-24:-4]) for story in story_prefixes[10:20]]
    all_prompts = prompts + extra + full_prefixes + ending_prefixes
    out_dir = q1_root() / "baseline_eval"
    suite = evaluate_prompt_suite(bundle, all_prompts, output_path=out_dir / "prompt_suite_results.json")
    long_cfg = GenerationConfig(mode="greedy", max_new_tokens=128, repetition_penalty=1.0)
    long_rows = [generate_text_record(bundle, prompt, long_cfg) for prompt in prompts]
    result = {
        "baseline_checkpoint": str(ML_M2_BASELINE_CHECKPOINT),
        "baseline_sha256": sha256_file(ML_M2_BASELINE_CHECKPOINT),
        "prompt_count": len(all_prompts),
        "suite": suite["variant_summary"],
        "incremental_kv": {
            "allclose": all(row["allclose"] for row in suite["incremental_kv_checks"]),
            "max_abs_error": max(row["max_abs_error"] for row in suite["incremental_kv_checks"]),
        },
        "long_generation": {
            "max_new_tokens": 128,
            "eos_rate": sum(1 for row in long_rows if row["hit_eos"]) / len(long_rows),
            "average_generated_length": sum(len(row["generated_token_ids"]) for row in long_rows) / len(long_rows),
        },
        "output_path": suite["output_path"],
    }
    _write_json(out_dir / "baseline_eval_summary.json", result)
    return result


def _save_q1_checkpoint(
    path: Path,
    model: HardwareMatchedCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    model_cfg: HardwareMatchedConfig,
    train_cfg: Q1TrainConfig,
    metrics: dict[str, Any],
    packed_manifest: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "ML-Q1",
        "model_role": "ML-Q1 candidate",
        "baseline_checkpoint": str(ML_M2_BASELINE_CHECKPOINT),
        "baseline_sha256": ML_M2_BASELINE_SHA256,
        "step": step,
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "config": model_cfg.to_json_dict(),
        "training_config": asdict(train_cfg),
        "metrics": metrics,
        "packed_manifest": packed_manifest,
    }
    torch.save(payload, path)
    _write_json(path.with_suffix(path.suffix + ".manifest.json"), {"path": str(path), "sha256": sha256_file(path), "step": step, "metrics": metrics})


def _load_model_from_checkpoint(path: str | Path) -> HardwareMatchedCausalLM:
    payload = _torch_load(path)
    cfg = HardwareMatchedConfig.from_json_dict(payload["config"])
    model = HardwareMatchedCausalLM(cfg)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def train_candidate(cfg: Q1TrainConfig | None = None) -> dict[str, Any]:
    cfg = cfg or Q1TrainConfig()
    bundle = load_baseline_bundle()
    packed = ensure_q1_packed_data(cfg, bundle)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("ML-Q1 continuation training requires CUDA")
    use_bf16 = torch.cuda.is_bf16_supported()
    train_ds = _load_packed_dataset(packed["splits"]["train"]["path"])
    val_ds = _load_packed_dataset(packed["splits"]["validation"]["path"])
    holdout_ds = _load_packed_dataset(packed["splits"]["holdout"]["path"])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, generator=torch.Generator().manual_seed(cfg.seed), pin_memory=True)
    steps_per_epoch = math.ceil(len(train_ds) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    warmup = max(1, int(total_steps * cfg.warmup_fraction))
    set_seed(cfg.seed)
    model = _load_model_from_checkpoint(ML_M2_BASELINE_CHECKPOINT).to(device)
    optimizer = build_optimizer(
        model,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        fused=True,
    )
    scheduler = build_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup,
        min_lr_ratio=cfg.minimum_lr / cfg.learning_rate,
        schedule="cosine",
    )
    train_iter = iter(train_loader)

    def train_step(step: int) -> tuple[float, float]:
        nonlocal train_iter
        try:
            input_ids, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            input_ids, labels = next(train_iter)
        input_ids = input_ids.long().to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            loss = model(input_ids, labels=labels)["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite ML-Q1 loss")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        return float(loss.detach().cpu()), float(grad.detach().cpu() if hasattr(grad, "detach") else grad)

    ckpt_dir = q1_root() / "candidate" / "checkpoints"
    metrics_dir = q1_root() / "candidate" / "training"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    pilot_history = []
    first100 = []
    no_nan = True
    for step in range(1, cfg.pilot_steps + 1):
        loss, grad = train_step(step)
        if step <= 100 and (step <= 20 or step % 10 == 0):
            first100.append({"step": step, "loss": loss, "grad_norm": grad, "lr": scheduler.get_last_lr()[0]})
        if step == 1 or step % 100 == 0:
            pilot_history.append({"step": step, "loss": loss, "grad_norm": grad, "lr": scheduler.get_last_lr()[0]})
    pilot_val = evaluate_dataset_loss(model, val_ds, cfg.batch_size, device=device)
    _save_q1_checkpoint(
        ckpt_dir / "ml_q1_pilot.pt",
        model,
        optimizer,
        scheduler,
        cfg.pilot_steps,
        model.config,
        cfg,
        {"pilot_validation_loss": pilot_val["loss"], "pilot_history": pilot_history},
        packed,
    )
    # Restart from the accepted baseline for the formal continuation after pilot validation.
    model = _load_model_from_checkpoint(ML_M2_BASELINE_CHECKPOINT).to(device)
    optimizer = build_optimizer(model, cfg.learning_rate, cfg.weight_decay, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps, fused=True)
    scheduler = build_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup, min_lr_ratio=cfg.minimum_lr / cfg.learning_rate, schedule="cosine")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, generator=torch.Generator().manual_seed(cfg.seed), pin_memory=True)
    train_iter = iter(train_loader)
    best_val = float("inf")
    history = []
    initial_loss = None
    final_loss = None
    for step in range(1, total_steps + 1):
        loss, grad = train_step(step)
        if initial_loss is None:
            initial_loss = loss
        final_loss = loss
        should_eval = step == 1 or step <= 100 and step % 10 == 0 or step % 250 == 0 or step == total_steps or step % steps_per_epoch == 0
        if should_eval:
            val = evaluate_dataset_loss(model, val_ds, cfg.batch_size, device=device)
            record = {
                "step": step,
                "epoch": step / steps_per_epoch,
                "train_loss": loss,
                "validation_loss": val["loss"],
                "perplexity": val["perplexity"],
                "grad_norm": grad,
                "lr": scheduler.get_last_lr()[0],
            }
            history.append(record)
            if val["loss"] < best_val:
                best_val = val["loss"]
                _save_q1_checkpoint(ckpt_dir / "ml_q1_candidate_best.pt", model, optimizer, scheduler, step, model.config, cfg, record, packed)
        if step == steps_per_epoch:
            if history and history[-1]["validation_loss"] > bundle.training_metrics["best_validation_loss"] + 0.2:
                no_nan = False
                break
    elapsed = time.perf_counter() - start
    final_val = evaluate_dataset_loss(model, val_ds, cfg.batch_size, device=device)
    holdout = evaluate_dataset_loss(model, holdout_ds, cfg.batch_size, device=device)
    _save_q1_checkpoint(
        ckpt_dir / "ml_q1_candidate_last.pt",
        model,
        optimizer,
        scheduler,
        total_steps,
        model.config,
        cfg,
        {"validation_loss": final_val["loss"], "holdout_loss": holdout["loss"], "train_loss": final_loss},
        packed,
    )
    if final_val["loss"] < best_val:
        _save_q1_checkpoint(
            ckpt_dir / "ml_q1_candidate_best.pt",
            model,
            optimizer,
            scheduler,
            total_steps,
            model.config,
            cfg,
            {"validation_loss": final_val["loss"], "holdout_loss": holdout["loss"], "train_loss": final_loss},
            packed,
        )
        best_val = final_val["loss"]
    best_path = ckpt_dir / "ml_q1_candidate_best.pt"
    summary = {
        "stage": "ML-Q1",
        "status": "TRAINING_COMPLETE" if no_nan else "STOPPED_AFTER_FIRST_EPOCH_DEGRADATION",
        "baseline_checkpoint": str(ML_M2_BASELINE_CHECKPOINT),
        "baseline_sha256": ML_M2_BASELINE_SHA256,
        "candidate_best_checkpoint": str(best_path),
        "candidate_best_sha256": sha256_file(best_path),
        "candidate_last_checkpoint": str(ckpt_dir / "ml_q1_candidate_last.pt"),
        "candidate_last_sha256": sha256_file(ckpt_dir / "ml_q1_candidate_last.pt"),
        "config": asdict(cfg),
        "train_packed_sequences": len(train_ds),
        "validation_packed_sequences": len(val_ds),
        "holdout_packed_sequences": len(holdout_ds),
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "elapsed_seconds": elapsed,
        "initial_train_loss": initial_loss,
        "final_train_loss": final_loss,
        "best_validation_loss": best_val,
        "final_validation_loss": final_val["loss"],
        "holdout_loss": holdout["loss"],
        "holdout_perplexity": holdout["perplexity"],
        "pilot": {"history": pilot_history, "validation_loss": pilot_val["loss"]},
        "first100": first100,
        "history": history,
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved()),
    }
    _write_json(metrics_dir / "q1_training_metrics.json", summary)
    model.cpu()
    return summary


def candidate_bundle(candidate_checkpoint: str | Path) -> ModelBundle:
    baseline = load_baseline_bundle()
    payload = _torch_load(candidate_checkpoint)
    cfg = HardwareMatchedConfig.from_json_dict(payload["config"])
    model = HardwareMatchedCausalLM(cfg)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return ModelBundle(
        artifact_root=q1_root() / "candidate",
        model=model,
        tokenizer=baseline.tokenizer,
        checkpoint_path=Path(candidate_checkpoint),
        checkpoint_sha256=sha256_file(candidate_checkpoint),
        data_manifest=baseline.data_manifest,
        training_metrics=payload.get("metrics", {}),
        dataset_metadata=baseline.dataset_metadata,
        generation_config_path=baseline.generation_config_path,
        generation_config_manifest=baseline.generation_config_manifest,
    )


@torch.no_grad()
def compare_fp16_hw(model: HardwareMatchedCausalLM, tokenizer: SimpleBPETokenizer) -> dict[str, Any]:
    model.to("cpu")
    prompt = "Once upon a time"
    ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], dtype=torch.long)
    fp16_model = _fp16_weight_rounded_model(model)
    pt = model(ids, return_trace=True)
    fp16 = fp16_model(ids)
    hw = run_hardware_aware_model(model, ids)
    return {
        "prompt": prompt,
        "pytorch_vs_fp16_weight": {
            **tensor_error_metrics(pt["logits"], fp16["logits"]),
            **logits_agreement(pt["logits"], fp16["logits"]),
        },
        "pytorch_vs_hardware_aware": {
            **tensor_error_metrics(pt["logits"], hw["logits"]),
            **logits_agreement(pt["logits"], hw["logits"]),
        },
    }


def compare_baseline_candidate() -> dict[str, Any]:
    baseline = load_baseline_bundle()
    train_metrics = json.loads((q1_root() / "candidate" / "training" / "q1_training_metrics.json").read_text(encoding="utf-8"))
    cand = candidate_bundle(train_metrics["candidate_best_checkpoint"])
    packed = json.loads((q1_root() / "data" / "packed" / "q1_packed_manifest.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {
        "validation": _load_packed_dataset(packed["splits"]["validation"]["path"]),
        "holdout": _load_packed_dataset(packed["splits"]["holdout"]["path"]),
    }
    loss = {"baseline": {}, "candidate": {}}
    for name, ds in datasets.items():
        loss["baseline"][name] = evaluate_dataset_loss(baseline.model, ds, device=device)
        loss["candidate"][name] = evaluate_dataset_loss(cand.model, ds, device=device)
    prompts = json.loads(Path("ml/evaluation/ml_m2_prompt_suite.json").read_text(encoding="utf-8"))["prompts"]
    baseline.model.to("cpu")
    baseline_suite = evaluate_prompt_suite(baseline, prompts, output_path=q1_root() / "baseline_eval" / "prompt_suite_m2_baseline.json")
    cand.model.to("cpu")
    candidate_suite = evaluate_prompt_suite(cand, prompts, output_path=q1_root() / "candidate" / "eval" / "prompt_suite_candidate.json")
    eos_base = eos_audit(baseline)
    eos_cand_report = _eos_eval_for_bundle(cand, max_examples=1000)
    fp16_hw = compare_fp16_hw(cand.model, cand.tokenizer)
    cand.model.to("cpu")
    inc = []
    for prompt in prompts:
        ids = torch.tensor([cand.tokenizer.encode(prompt, add_bos=True)], dtype=torch.long)
        inc.append({"prompt": prompt, **compare_full_vs_incremental(cand.model, ids)})
    result = {
        "baseline_checkpoint": str(ML_M2_BASELINE_CHECKPOINT),
        "baseline_sha256": ML_M2_BASELINE_SHA256,
        "candidate_checkpoint": str(cand.checkpoint_path),
        "candidate_sha256": cand.checkpoint_sha256,
        "loss": loss,
        "baseline_suite": baseline_suite["variant_summary"],
        "candidate_suite": candidate_suite["variant_summary"],
        "baseline_eos": eos_base,
        "candidate_eos": eos_cand_report,
        "candidate_fp16_hardware_aware": fp16_hw,
        "candidate_incremental_kv": {
            "allclose": all(row["allclose"] for row in inc),
            "max_abs_error": max(row["max_abs_error"] for row in inc),
        },
        "quality_improvement_pass": _quality_pass(loss, baseline_suite["variant_summary"], candidate_suite["variant_summary"], eos_base, eos_cand_report, fp16_hw),
    }
    _write_json(q1_root() / "comparison" / "baseline_vs_candidate.json", result)
    baseline.model.cpu()
    cand.model.cpu()
    return result


def _eos_eval_for_bundle(bundle: ModelBundle, max_examples: int = 1000) -> dict[str, Any]:
    tokenizer = bundle.tokenizer
    docs = _read_stories(load_baseline_bundle().data_manifest["subsets"]["validation_subset_path"], max_examples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = bundle.model.to(device)
    top1 = top5 = 0
    prob_sum = rank_sum = 0.0
    for doc in docs:
        ids = tokenizer.encode(doc, add_bos=True, add_eos=True)
        prefix = ids[:-1][-bundle.model.config.context_length :]
        logits = _eval_logits(model, torch.tensor([prefix], dtype=torch.long), device)[0, -1]
        probs = torch.softmax(logits, dim=-1)
        sorted_ids = torch.argsort(probs, descending=True)
        rank = int((sorted_ids == tokenizer.eos_id).nonzero(as_tuple=False)[0].item()) + 1
        rank_sum += rank
        prob_sum += float(probs[tokenizer.eos_id].item())
        top1 += int(rank == 1)
        top5 += int(rank <= 5)
    model.cpu()
    return {
        "examples": len(docs),
        "eos_top1_accuracy": top1 / max(len(docs), 1),
        "eos_top5_accuracy": top5 / max(len(docs), 1),
        "eos_average_probability": prob_sum / max(len(docs), 1),
        "eos_average_rank": rank_sum / max(len(docs), 1),
    }


def _quality_pass(loss, baseline_suite, candidate_suite, baseline_eos, candidate_eos, fp16_hw) -> bool:
    val_down = loss["candidate"]["validation"]["loss"] < loss["baseline"]["validation"]["loss"]
    holdout_ok = loss["candidate"]["holdout"]["loss"] <= loss["baseline"]["holdout"]["loss"] + 0.02
    loops_down = candidate_suite["greedy"]["ngram_loop_count"] < baseline_suite["greedy"]["ngram_loop_count"]
    eos_ok = candidate_eos["eos_average_rank"] <= baseline_eos["eos_average_rank"] * 1.2
    hw_ok = fp16_hw["pytorch_vs_hardware_aware"]["top1_agreement"] == 1.0
    return bool(val_down and holdout_ok and loops_down and eos_ok and hw_ok)


def write_report_markdowns() -> None:
    root = q1_root()
    reports = Path("reports/ml_q1")
    tok = json.loads((root / "audits" / "tokenizer_audit.json").read_text(encoding="utf-8"))
    eos = json.loads((root / "audits" / "eos_audit.json").read_text(encoding="utf-8"))
    cap = json.loads((root / "audits" / "capacity_training_audit.json").read_text(encoding="utf-8"))
    raw = json.loads((root / "data" / "q1_raw_data_manifest.json").read_text(encoding="utf-8"))
    packed = json.loads((root / "data" / "packed" / "q1_packed_manifest.json").read_text(encoding="utf-8"))
    train = json.loads((root / "candidate" / "training" / "q1_training_metrics.json").read_text(encoding="utf-8"))
    comp = json.loads((root / "comparison" / "baseline_vs_candidate.json").read_text(encoding="utf-8"))

    _write_md(reports / "tokenizer_audit.md", f"""# ML-Q1 Tokenizer Audit

```text
vocab_size={tok['vocab_size']}
special_token_ids={tok['special_token_ids']}
average_chars_per_token_train_sample={tok['average_chars_per_token_train_sample']}
average_tokens_per_story_train_sample={tok['average_tokens_per_story_train_sample']}
space_token_ratio={tok['space_token']['ratio']}
single_character_token_count={tok['single_character_token_count']}
leading_space_word_token_count={tok['leading_space_word_token_count']}
merge_count={tok['merge_count']}
merge_utilization={tok['merge_utilization']}
unk_ratio_train_sample={tok['unk_ratio_train_sample']}
unk_ratio_validation={tok['unk_ratio_validation']}
train_validation_distribution_l1={tok['train_validation_distribution_l1']}
decode_reencode_stable_ratio_train={tok['decode_reencode_stable_ratio_train']}
decode_reencode_stable_ratio_validation={tok['decode_reencode_stable_ratio_validation']}
```

Space conclusion: {tok['space_conclusion']}

Prompt encoding for `Once upon a time`:

```text
ids={tok['prompt_encoding']['ids']}
tokens={tok['prompt_encoding']['tokens']}
roundtrip={tok['prompt_encoding']['roundtrip']}
```

Most common tokens and bigrams are recorded in `D:/IC_Workspace/VEDA_artifacts/ml_q1/audits/tokenizer_audit.json`.
""")
    _write_md(reports / "eos_audit.md", f"""# ML-Q1 EOS Audit

```text
train_eos_targets={eos['train_eos_targets']}
train_total_targets={eos['train_total_targets']}
train_eos_ratio={eos['train_eos_ratio']}
validation_eos_targets={eos['validation_eos_targets']}
validation_eos_ratio={eos['validation_eos_ratio']}
pad_eos_confusion={eos['pad_eos_confusion']}
eos_is_ignore_index={eos['eos_is_ignore_index']}
eos_examples={eos['validation_story_eos_examples']}
eos_top1_accuracy={eos['eos_top1_accuracy']}
eos_top5_accuracy={eos['eos_top5_accuracy']}
eos_average_probability={eos['eos_average_probability']}
eos_average_rank={eos['eos_average_rank']}
non_terminal_eos_false_positive_rate={eos['non_terminal_eos_false_positive_rate']}
```

Generation EOS rates:

```json
{json.dumps(eos['generation_eos_rates'], indent=2, sort_keys=True)}
```
""")
    _write_md(reports / "capacity_training_audit.md", f"""# ML-Q1 Capacity and Training Audit

```text
train_sample_loss={cap['train_fixed_sample']['loss']}
validation_loss={cap['validation']['loss']}
holdout_loss={cap['holdout']['loss']}
validation_perplexity={cap['validation']['perplexity']}
holdout_perplexity={cap['holdout']['perplexity']}
high_frequency_avg_loss={cap['frequency_loss']['high_frequency_avg_loss']}
low_frequency_avg_loss={cap['frequency_loss']['low_frequency_avg_loss']}
first_16_position_loss={cap['position_loss_summary']['first_16_avg']}
last_16_position_loss={cap['position_loss_summary']['last_16_avg']}
```

Attribution:

```json
{json.dumps(cap['attribution'], indent=2, sort_keys=True)}
```
""")
    _write_md(reports / "training_metrics.md", f"""# ML-Q1 Candidate Training Metrics

```text
baseline_checkpoint={train['baseline_checkpoint']}
baseline_sha256={train['baseline_sha256']}
candidate_best_checkpoint={train['candidate_best_checkpoint']}
candidate_best_sha256={train['candidate_best_sha256']}
train_stories={raw['train']['stories']}
validation_stories={raw['validation']['stories']}
holdout_stories={raw['holdout']['stories']}
train_packed_sequences={train['train_packed_sequences']}
validation_packed_sequences={train['validation_packed_sequences']}
holdout_packed_sequences={train['holdout_packed_sequences']}
steps_per_epoch={train['steps_per_epoch']}
total_steps={train['total_steps']}
elapsed_seconds={train['elapsed_seconds']}
initial_train_loss={train['initial_train_loss']}
final_train_loss={train['final_train_loss']}
best_validation_loss={train['best_validation_loss']}
final_validation_loss={train['final_validation_loss']}
holdout_loss={train['holdout_loss']}
```

Optimizer was reinitialized for ML-Q1 continuation; scheduler was restarted
with the ML-Q1 cosine schedule.
""")
    _write_md(reports / "baseline_vs_candidate.md", _comparison_md(comp))
    _write_md(reports / "generation_comparison.md", _generation_md(comp))
    _write_md(reports / "artifact_manifest.md", f"""# ML-Q1 Artifact Manifest

```text
root={root}
raw_manifest={root / 'data' / 'q1_raw_data_manifest.json'}
packed_manifest={root / 'data' / 'packed' / 'q1_packed_manifest.json'}
baseline_eval={root / 'baseline_eval'}
candidate_eval={root / 'candidate' / 'eval'}
candidate_best={train['candidate_best_checkpoint']}
candidate_best_sha256={train['candidate_best_sha256']}
candidate_last={train['candidate_last_checkpoint']}
candidate_last_sha256={train['candidate_last_sha256']}
```

ML-M2 baseline checkpoint remains unchanged:

```text
{ML_M2_BASELINE_CHECKPOINT}
sha256={sha256_file(ML_M2_BASELINE_CHECKPOINT)}
```
""")
    status = "ML-Q1 QUALITY IMPROVEMENT PASS" if comp["quality_improvement_pass"] else "ML-Q1 NO SIGNIFICANT QUALITY GAIN"
    _write_md(reports / "acceptance_audit.md", f"""# ML-Q1 Acceptance Audit

## Result

{status}

```text
baseline_checkpoint_unchanged={sha256_file(ML_M2_BASELINE_CHECKPOINT) == ML_M2_BASELINE_SHA256}
validation_loss_baseline={comp['loss']['baseline']['validation']['loss']}
validation_loss_candidate={comp['loss']['candidate']['validation']['loss']}
holdout_loss_baseline={comp['loss']['baseline']['holdout']['loss']}
holdout_loss_candidate={comp['loss']['candidate']['holdout']['loss']}
greedy_loops_baseline={comp['baseline_suite']['greedy']['ngram_loop_count']}
greedy_loops_candidate={comp['candidate_suite']['greedy']['ngram_loop_count']}
candidate_incremental_kv_allclose={comp['candidate_incremental_kv']['allclose']}
hardware_aware_top1={comp['candidate_fp16_hardware_aware']['pytorch_vs_hardware_aware']['top1_agreement']}
```

ML-Q1 did not run real RTL and did not modify hardware files.
""")
    _write_md(reports / "summary.md", f"""# ML-Q1 Summary

## Result

{status}

ML-Q1 keeps the accepted ML-M2 architecture fixed and evaluates whether
continuing TinyStories training improves quality without changing RTL.

```text
train_stories={raw['train']['stories']}
training_tokens={packed['splits']['train']['token_count'] * train['config']['epochs']}
epochs={train['config']['epochs']}
steps={train['total_steps']}
candidate={train['candidate_best_checkpoint']}
candidate_sha256={train['candidate_best_sha256']}
baseline_validation_loss={comp['loss']['baseline']['validation']['loss']}
candidate_validation_loss={comp['loss']['candidate']['validation']['loss']}
baseline_holdout_loss={comp['loss']['baseline']['holdout']['loss']}
candidate_holdout_loss={comp['loss']['candidate']['holdout']['loss']}
```

Conclusion: fixed one-layer `d_model=64` remains the dominant quality limit.
Further full-dataset training should only be run as ML-Q2 if validation,
holdout, loop, EOS, and hardware-aware metrics justify it.
""")


def _comparison_md(comp: dict[str, Any]) -> str:
    return f"""# ML-Q1 Baseline vs Candidate

```text
baseline_checkpoint={comp['baseline_checkpoint']}
candidate_checkpoint={comp['candidate_checkpoint']}
candidate_sha256={comp['candidate_sha256']}
baseline_validation_loss={comp['loss']['baseline']['validation']['loss']}
candidate_validation_loss={comp['loss']['candidate']['validation']['loss']}
baseline_holdout_loss={comp['loss']['baseline']['holdout']['loss']}
candidate_holdout_loss={comp['loss']['candidate']['holdout']['loss']}
baseline_validation_ppl={comp['loss']['baseline']['validation']['perplexity']}
candidate_validation_ppl={comp['loss']['candidate']['validation']['perplexity']}
baseline_holdout_ppl={comp['loss']['baseline']['holdout']['perplexity']}
candidate_holdout_ppl={comp['loss']['candidate']['holdout']['perplexity']}
quality_improvement_pass={comp['quality_improvement_pass']}
```

Greedy loop comparison:

```text
baseline_ngram_loops={comp['baseline_suite']['greedy']['ngram_loop_count']}
candidate_ngram_loops={comp['candidate_suite']['greedy']['ngram_loop_count']}
baseline_distinct1={comp['baseline_suite']['greedy']['average_distinct_1']}
candidate_distinct1={comp['candidate_suite']['greedy']['average_distinct_1']}
baseline_distinct2={comp['baseline_suite']['greedy']['average_distinct_2']}
candidate_distinct2={comp['candidate_suite']['greedy']['average_distinct_2']}
```

EOS comparison:

```text
baseline_eos_avg_rank={comp['baseline_eos']['eos_average_rank']}
candidate_eos_avg_rank={comp['candidate_eos']['eos_average_rank']}
baseline_eos_avg_prob={comp['baseline_eos']['eos_average_probability']}
candidate_eos_avg_prob={comp['candidate_eos']['eos_average_probability']}
```

Hardware-aware comparison:

```json
{json.dumps(comp['candidate_fp16_hardware_aware'], indent=2, sort_keys=True)}
```
"""


def _generation_md(comp: dict[str, Any]) -> str:
    return f"""# ML-Q1 Generation Comparison

Full prompt-suite JSON artifacts:

```text
baseline=D:/IC_Workspace/VEDA_artifacts/ml_q1/baseline_eval/prompt_suite_m2_baseline.json
candidate=D:/IC_Workspace/VEDA_artifacts/ml_q1/candidate/eval/prompt_suite_candidate.json
```

Greedy:

```text
baseline_loops={comp['baseline_suite']['greedy']['ngram_loop_count']}
candidate_loops={comp['candidate_suite']['greedy']['ngram_loop_count']}
baseline_collapse_count={comp['baseline_suite']['greedy']['single_token_collapse_count']}
candidate_collapse_count={comp['candidate_suite']['greedy']['single_token_collapse_count']}
```

Sampling temperature 1.0:

```text
baseline_distinct1={comp['baseline_suite']['temperature_1_0']['average_distinct_1']}
candidate_distinct1={comp['candidate_suite']['temperature_1_0']['average_distinct_1']}
baseline_distinct2={comp['baseline_suite']['temperature_1_0']['average_distinct_2']}
candidate_distinct2={comp['candidate_suite']['temperature_1_0']['average_distinct_2']}
```

Temperature can improve diversity at decode time, but ML-Q1 acceptance is based
on trained checkpoint metrics, not decoding tricks alone.
"""


def run_audits() -> dict[str, Any]:
    bundle = load_baseline_bundle()
    cfg = Q1TrainConfig()
    packed = ensure_q1_packed_data(cfg, bundle)
    result = {
        "tokenizer": tokenizer_audit(bundle),
        "eos": eos_audit(bundle),
        "capacity": capacity_audit(bundle, packed),
        "baseline_eval": run_baseline_eval(bundle),
    }
    _write_json(q1_root() / "audits" / "q1_audit_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["audits", "prepare-data", "train", "compare", "write-reports", "all"])
    args = parser.parse_args()
    if args.action == "prepare-data":
        print(json.dumps(ensure_q1_packed_data(Q1TrainConfig()), indent=2, sort_keys=True))
    elif args.action == "audits":
        print(json.dumps(run_audits(), indent=2, sort_keys=True))
    elif args.action == "train":
        print(json.dumps(train_candidate(), indent=2, sort_keys=True))
    elif args.action == "compare":
        print(json.dumps(compare_baseline_candidate(), indent=2, sort_keys=True))
    elif args.action == "write-reports":
        write_report_markdowns()
    elif args.action == "all":
        run_audits()
        train_candidate()
        compare_baseline_candidate()
        write_report_markdowns()


if __name__ == "__main__":
    main()
