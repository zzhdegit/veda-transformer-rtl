#!/usr/bin/env python3
"""Validate Stage M1 generated metadata and weight-free artifact policy."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


FORBIDDEN_SUFFIXES = {
    ".bin",
    ".safetensors",
    ".pt",
    ".pth",
    ".ckpt",
    ".gguf",
    ".onnx",
    ".tflite",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_rows(rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    if len(rows) < 10:
        errors.append(f"expected at least 10 candidates, found {len(rows)}")
    detail_count = sum(str(row.get("detail", "")).lower() == "true" for row in rows)
    if detail_count < 5:
        errors.append(f"expected at least 5 detailed candidates, found {detail_count}")
    for row in rows:
        model_id = row.get("model_id", "")
        if not model_id:
            errors.append("row missing model_id")
        if not row.get("revision") and model_id != "meta-llama/Llama-2-7b-hf":
            errors.append(f"{model_id}: missing revision")
        if not row.get("license"):
            print(f"notice: {model_id}: license not declared in HF card/API", file=sys.stderr)
        for key in ("model_type", "hidden_size", "num_attention_heads", "head_dim"):
            if not row.get(key) and model_id != "meta-llama/Llama-2-7b-hf":
                errors.append(f"{model_id}: missing {key}")
    return errors


def validate_generated_dir(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return errors
    for child in path.rglob("*"):
        if child.is_file() and child.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden weight/cache-like artifact: {child}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="reports/model_selection/generated/candidate_matrix.csv")
    parser.add_argument("--generated-dir", default="reports/model_selection/generated")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"metadata CSV missing: {csv_path}", file=sys.stderr)
        return 1

    errors = validate_rows(load_rows(csv_path))
    errors.extend(validate_generated_dir(Path(args.generated_dir)))
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("metadata validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
