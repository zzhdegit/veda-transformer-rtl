#!/usr/bin/env python3
"""Estimate Transformer KV cache size from inspected model metadata."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def parse_int(value: str | int | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def estimate(row: dict[str, str]) -> dict[str, str | int]:
    layers = parse_int(row.get("layers"))
    kv_heads = parse_int(row.get("num_key_value_heads"))
    head_dim = parse_int(row.get("head_dim"))
    context = parse_int(row.get("max_position_embeddings"))
    if not (layers and kv_heads and head_dim):
        raise ValueError(f"missing KV dimensions for {row.get('model_id')}")

    per_token_per_layer_fp16 = 2 * kv_heads * head_dim * 2
    per_token_per_layer_fp32 = 2 * kv_heads * head_dim * 4
    context = context or 0
    return {
        "model_id": row["model_id"],
        "layers": layers,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "context": context,
        "kv_per_token_per_layer_fp16_bytes": per_token_per_layer_fp16,
        "kv_per_token_per_layer_fp32_bytes": per_token_per_layer_fp32,
        "kv_per_token_all_layers_fp16_bytes": per_token_per_layer_fp16 * layers,
        "kv_per_token_all_layers_fp32_bytes": per_token_per_layer_fp32 * layers,
        "full_context_fp16_bytes": per_token_per_layer_fp16 * layers * context,
        "full_context_fp32_bytes": per_token_per_layer_fp32 * layers * context,
    }


def format_mib(value: int) -> str:
    return f"{value / (1024 * 1024):.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="reports/model_selection/generated/candidate_matrix.csv")
    parser.add_argument("--out", help="Optional CSV output path.")
    args = parser.parse_args()

    with open(args.csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    estimates: list[dict[str, str | int]] = []
    for row in rows:
        try:
            estimates.append(estimate(row))
        except ValueError as exc:
            print(f"warning: {exc}", file=sys.stderr)

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(estimates[0]))
            writer.writeheader()
            writer.writerows(estimates)

    for item in estimates:
        print(
            f"{item['model_id']}: per-token/layer FP16={item['kv_per_token_per_layer_fp16_bytes']} B, "
            f"all-layer/token FP16={format_mib(int(item['kv_per_token_all_layers_fp16_bytes']))} MiB, "
            f"full-context FP16={format_mib(int(item['full_context_fp16_bytes']))} MiB"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
