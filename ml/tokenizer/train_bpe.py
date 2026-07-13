"""Train the deterministic ML-M2 BPE tokenizer."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.data.dataset_hash import sha256_file, sha256_text
from ml.data.dataset_manifest import artifact_root
from ml.data.fixtures import fixture_text
from ml.tokenizer.load_tokenizer import SimpleBPETokenizer
from ml.tokenizer.tokenizer_manifest import TokenizerManifest, write_tokenizer_manifest


def train_tokenizer_from_texts(
    texts: list[str],
    output_dir: str | Path,
    vocab_size: int = 2048,
    seed: int = 20260713,
    source_dataset: str = "builtin_fixture",
    source_sha256: str | None = None,
) -> tuple[SimpleBPETokenizer, TokenizerManifest]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = SimpleBPETokenizer.train(texts, vocab_size=vocab_size)
    tokenizer_path = out / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    stats = tokenizer.stats(texts)
    tokenizer_sha = sha256_file(tokenizer_path)
    manifest = TokenizerManifest(
        tokenizer_type="simple_bpe",
        vocab_size=len(tokenizer.vocab),
        requested_vocab_size=vocab_size,
        merge_count=len(tokenizer.merges),
        source_dataset=source_dataset,
        source_sha256=source_sha256 or sha256_text("\n".join(texts)),
        seed=seed,
        special_tokens={
            "pad": tokenizer.pad_id,
            "bos": tokenizer.bos_id,
            "eos": tokenizer.eos_id,
            "unk": tokenizer.unk_id,
        },
        tokenizer_json=str(tokenizer_path),
        tokenizer_sha256=tokenizer_sha,
        average_encoded_length=stats.average_encoded_length,
        unk_tokens=stats.unk_tokens,
        total_tokens=stats.total_tokens,
    )
    write_tokenizer_manifest(manifest, out / "tokenizer_manifest.json")
    return tokenizer, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", help="UTF-8 text file. May be repeated.")
    parser.add_argument("--output-dir", default=str(artifact_root() / "tokenizers" / "ml_m2"))
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()

    if args.input:
        texts = [Path(path).read_text(encoding="utf-8", errors="replace") for path in args.input]
        source_dataset = ",".join(args.input)
        source_sha = sha256_text("\n".join(texts))
    else:
        texts = [fixture_text()]
        source_dataset = "builtin_fixture"
        source_sha = sha256_text(texts[0])
    _, manifest = train_tokenizer_from_texts(
        texts,
        args.output_dir,
        vocab_size=args.vocab_size,
        seed=args.seed,
        source_dataset=source_dataset,
        source_sha256=source_sha,
    )
    print(f"tokenizer={manifest.tokenizer_json}")
    print(f"vocab_size={manifest.vocab_size} merges={manifest.merge_count}")
    print(f"avg_len={manifest.average_encoded_length:.2f} unk={manifest.unk_tokens}/{manifest.total_tokens}")


if __name__ == "__main__":
    main()

