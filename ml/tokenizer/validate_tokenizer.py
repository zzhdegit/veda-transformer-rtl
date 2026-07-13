"""Validate a saved ML-M2 tokenizer."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.tokenizer.load_tokenizer import SimpleBPETokenizer


def validate_tokenizer(path: str | Path) -> dict[str, int]:
    tokenizer = SimpleBPETokenizer.load(path)
    required = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    for token, expected in required.items():
        actual = tokenizer.vocab.get(token)
        if actual != expected:
            raise ValueError(f"{token} expected id {expected}, got {actual}")
    text = "A small test."
    encoded = tokenizer.encode(text, add_bos=True, add_eos=True)
    decoded = tokenizer.decode(encoded)
    if decoded != text:
        raise ValueError(f"decode mismatch: {decoded!r} != {text!r}")
    return {"vocab_size": len(tokenizer.vocab), "merge_count": len(tokenizer.merges)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer_json")
    args = parser.parse_args()
    result = validate_tokenizer(args.tokenizer_json)
    print(f"tokenizer validation passed vocab_size={result['vocab_size']} merges={result['merge_count']}")


if __name__ == "__main__":
    main()

