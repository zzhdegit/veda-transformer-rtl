"""Formal TinyStories artifact preparation for ML-M2."""

from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import torch

from ml.architecture.config import HardwareMatchedConfig
from ml.data.dataset_hash import sha256_file
from ml.data.dataset_manifest import ACCESS_DATE, artifact_root, data_root
from ml.data.sequence_builder import build_lm_sequences
from ml.data.tinystories_loader import (
    TINYSTORIES_CARD_URL,
    TINYSTORIES_LICENSE,
    TINYSTORIES_REVISION,
    TINYSTORIES_TRAIN_URL,
    TINYSTORIES_VALID_URL,
    split_tinystories_text,
)
from ml.inference.prompt_suite import formal_prompts
from ml.tokenizer.train_bpe import train_tokenizer_from_texts


TINYSTORIES_SOURCE_COMMIT = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"


def _default_formal_root() -> Path:
    return artifact_root() / "formal"


def _read_stories(path: Path, limit: int | None = None) -> list[str]:
    stories = split_tinystories_text(path.read_text(encoding="utf-8", errors="replace"))
    return stories if limit is None else stories[:limit]


def _write_stories(path: Path, stories: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    marker = "\n<|endoftext|>\n"
    path.write_text(marker.join(stories) + marker, encoding="utf-8")


def stream_tinystories_subset(url: str, output_path: str | Path, target_stories: int) -> dict:
    """Stream enough TinyStories records from the official text file.

    The full training file is large, so this function stops once the requested
    deterministic prefix has been written under the artifact data directory.
    """

    target = Path(output_path)
    if target.exists():
        existing = _read_stories(target)
        if len(existing) >= target_stories:
            selected = existing[:target_stories]
            _write_stories(target, selected)
            return {
                "path": str(target),
                "stories": len(selected),
                "bytes_read": target.stat().st_size,
                "elapsed_seconds": 0.0,
                "reused_existing": True,
            }

    target.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "veda-ml-m2-formal/1.0"})
    marker = "<|endoftext|>"
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buffer = ""
    stories: list[str] = []
    bytes_read = 0
    start = time.perf_counter()
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with urlopen(request, timeout=120) as response:
                while len(stories) < target_stories:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    bytes_read += len(chunk)
                    buffer += decoder.decode(chunk)
                    parts = buffer.split(marker)
                    buffer = parts[-1]
                    for part in parts[:-1]:
                        story = part.strip()
                        if story:
                            stories.append(story)
                            if len(stories) >= target_stories:
                                break
            break
        except (OSError, URLError) as exc:
            last_error = exc
            if attempt == 3:
                raise
            stories = []
            buffer = ""
            bytes_read = 0
            time.sleep(5 * attempt)
    if len(stories) < target_stories and buffer.strip():
        stories.append(buffer.strip())
    selected = stories[:target_stories]
    _write_stories(target, selected)
    return {
        "path": str(target),
        "stories": len(selected),
        "bytes_read": bytes_read,
        "elapsed_seconds": time.perf_counter() - start,
        "reused_existing": False,
    }


def _encode_documents(tokenizer, docs: list[str]) -> list[int]:
    ids: list[int] = []
    for doc in docs:
        ids.extend(tokenizer.encode(doc, add_bos=True, add_eos=True))
    return ids


def _save_sequence_tensor(path: Path, input_ids: list[list[int]], labels: list[list[int]]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    input_tensor = torch.tensor(input_ids, dtype=torch.long)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    torch.save({"input_ids": input_tensor, "labels": label_tensor}, path)
    non_ignored = int((label_tensor != -100).sum().item())
    total_labels = int(label_tensor.numel())
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "shape": list(input_tensor.shape),
        "non_ignored_labels": non_ignored,
        "total_labels": total_labels,
        "pad_label_ratio": 1.0 - (non_ignored / total_labels if total_labels else 0.0),
    }


def _overlap_count(train_docs: list[str], validation_docs: list[str]) -> int:
    train_hashes = {hashlib.sha256(doc.encode("utf-8", errors="replace")).hexdigest() for doc in train_docs}
    return sum(
        1
        for doc in validation_docs
        if hashlib.sha256(doc.encode("utf-8", errors="replace")).hexdigest() in train_hashes
    )


def prepare_formal_data(
    output_dir: str | Path | None = None,
    train_stories: int = 100000,
    validation_stories: int = 10000,
    tokenizer_docs: int | None = None,
    vocab_size: int = 2048,
    context_length: int = 128,
    seed: int = 20260713,
) -> dict:
    out = Path(output_dir) if output_dir else _default_formal_root()
    data_dir = data_root() / "tinystories_ml_m2"
    formal_data_dir = out / "data"
    tokenizer_dir = out / "tokenizer"
    formal_data_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    train_subset = data_dir / f"TinyStories-train-prefix-{train_stories}.txt"
    valid_subset = data_dir / f"TinyStories-valid-prefix-{validation_stories}.txt"
    train_stream = stream_tinystories_subset(TINYSTORIES_TRAIN_URL, train_subset, train_stories)
    valid_stream = stream_tinystories_subset(TINYSTORIES_VALID_URL, valid_subset, validation_stories)
    train_docs = _read_stories(train_subset, train_stories)
    validation_docs = _read_stories(valid_subset, validation_stories)

    tokenizer_training_docs = train_docs if tokenizer_docs is None else train_docs[:tokenizer_docs]
    tokenizer, tok_manifest = train_tokenizer_from_texts(
        tokenizer_training_docs,
        tokenizer_dir,
        vocab_size=vocab_size,
        seed=seed,
        source_dataset="roneneldan/TinyStories train split prefix",
        source_sha256=sha256_file(train_subset),
    )

    train_ids = _encode_documents(tokenizer, train_docs)
    validation_ids = _encode_documents(tokenizer, validation_docs)
    train_batch = build_lm_sequences(train_ids, context_length=context_length, pad_id=tokenizer.pad_id, stride=context_length)
    validation_batch = build_lm_sequences(
        validation_ids,
        context_length=context_length,
        pad_id=tokenizer.pad_id,
        stride=context_length,
    )
    train_tensor = _save_sequence_tensor(formal_data_dir / "packed_train.pt", train_batch.input_ids, train_batch.labels)
    validation_tensor = _save_sequence_tensor(
        formal_data_dir / "packed_validation.pt",
        validation_batch.input_ids,
        validation_batch.labels,
    )

    token_total = len(train_ids)
    packed_capacity = max(len(train_batch.labels) * context_length, 1)
    unk_count = sum(1 for token_id in train_ids if token_id == tokenizer.unk_id)
    prompt_path = formal_data_dir / "test_prompts.json"
    prompt_path.write_text(json.dumps({"prompts": formal_prompts()}, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "stage": "ML-M2 Formal",
        "created_at": ACCESS_DATE,
        "dataset": {
            "name": "TinyStories",
            "source_url": TINYSTORIES_CARD_URL,
            "train_url": TINYSTORIES_TRAIN_URL,
            "validation_url": TINYSTORIES_VALID_URL,
            "revision": TINYSTORIES_REVISION,
            "source_commit": TINYSTORIES_SOURCE_COMMIT,
            "license": TINYSTORIES_LICENSE,
            "access_date": ACCESS_DATE,
        },
        "subsets": {
            "train_stories": len(train_docs),
            "validation_stories": len(validation_docs),
            "target_train_stories": train_stories,
            "target_validation_stories": validation_stories,
            "train_raw_characters": sum(len(doc) for doc in train_docs),
            "validation_raw_characters": sum(len(doc) for doc in validation_docs),
            "train_stream": train_stream,
            "validation_stream": valid_stream,
            "train_subset_path": str(train_subset),
            "validation_subset_path": str(valid_subset),
            "train_subset_sha256": sha256_file(train_subset),
            "validation_subset_sha256": sha256_file(valid_subset),
            "train_validation_overlap_count": _overlap_count(train_docs, validation_docs),
        },
        "tokenizer": {
            "type": "simple_bpe",
            "vocab_size": len(tokenizer.vocab),
            "target_vocab_size": vocab_size,
            "special_tokens": {"PAD": 0, "BOS": 1, "EOS": 2, "UNK": 3},
            "training_docs": len(tokenizer_training_docs),
            "manifest": tok_manifest.__dict__,
            "tokenizer_json": str(tokenizer_dir / "tokenizer.json"),
            "tokenizer_sha256": sha256_file(tokenizer_dir / "tokenizer.json"),
            "unk_count": unk_count,
            "unk_ratio": unk_count / max(token_total, 1),
            "average_tokens_per_story": token_total / max(len(train_docs), 1),
        },
        "packing": {
            "context_length": context_length,
            "train_token_count": token_total,
            "validation_token_count": len(validation_ids),
            "train_packed_sequences": len(train_batch.input_ids),
            "validation_packed_sequences": len(validation_batch.input_ids),
            "packing_utilization": min(1.0, token_total / packed_capacity),
            "train_tensor": train_tensor,
            "validation_tensor": validation_tensor,
        },
        "test_prompts": {"path": str(prompt_path), "sha256": sha256_file(prompt_path)},
        "environment": {
            "veda_ml_data_root": str(data_root()),
            "veda_ml_artifact_root": str(artifact_root()),
            "veda_hf_cache": os.environ.get("VEDA_HF_CACHE", ""),
        },
    }
    manifest_path = formal_data_dir / "formal_data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(_default_formal_root()))
    parser.add_argument("--train-stories", type=int, default=100000)
    parser.add_argument("--validation-stories", type=int, default=10000)
    parser.add_argument("--tokenizer-docs", type=int, default=0, help="0 means use all training stories")
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()
    cfg = HardwareMatchedConfig(vocab_size=args.vocab_size, context_length=args.context_length, seed=args.seed)
    manifest = prepare_formal_data(
        output_dir=args.output_dir,
        train_stories=args.train_stories,
        validation_stories=args.validation_stories,
        tokenizer_docs=None if args.tokenizer_docs == 0 else args.tokenizer_docs,
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        seed=cfg.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
