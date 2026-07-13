"""Tokenizer manifest helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TokenizerManifest:
    tokenizer_type: str
    vocab_size: int
    requested_vocab_size: int
    merge_count: int
    source_dataset: str
    source_sha256: str
    seed: int
    special_tokens: dict[str, int]
    tokenizer_json: str
    tokenizer_sha256: str
    average_encoded_length: float
    unk_tokens: int
    total_tokens: int


def write_tokenizer_manifest(manifest: TokenizerManifest, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_tokenizer_manifest(path: str | Path) -> TokenizerManifest:
    return TokenizerManifest(**json.loads(Path(path).read_text(encoding="utf-8")))

