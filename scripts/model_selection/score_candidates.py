#!/usr/bin/env python3
"""Score Stage M1 model candidates with explicit, reviewable components."""

from __future__ import annotations

import argparse
import csv
import sys


WEIGHTS = {
    "rtl_compatibility": 25,
    "simulation_scale": 20,
    "availability": 10,
    "architecture_clarity": 10,
    "license_clarity": 10,
    "kv_eviction_value": 10,
    "veda_closeness": 10,
    "tool_support": 5,
}


SCORES: dict[str, dict[str, int | str]] = {
    "facebook/opt-125m": {
        "rtl_compatibility": 16,
        "simulation_scale": 13,
        "availability": 8,
        "architecture_clarity": 9,
        "license_clarity": 4,
        "kv_eviction_value": 5,
        "veda_closeness": 5,
        "tool_support": 5,
        "rationale": "ReLU, Pre-LN, MHA, 4D FFN; but LayerNorm, learned positions and bias require hardware or wrapper changes.",
    },
    "roneneldan/TinyStories-1M": {
        "rtl_compatibility": 9,
        "simulation_scale": 20,
        "availability": 8,
        "architecture_clarity": 8,
        "license_clarity": 2,
        "kv_eviction_value": 3,
        "veda_closeness": 3,
        "tool_support": 4,
        "rationale": "Smallest trained checkpoint; architecture uses GPT-Neo LayerNorm, GELU and local attention.",
    },
    "HuggingFaceTB/SmolLM2-135M": {
        "rtl_compatibility": 11,
        "simulation_scale": 15,
        "availability": 10,
        "architecture_clarity": 10,
        "license_clarity": 10,
        "kv_eviction_value": 9,
        "veda_closeness": 8,
        "tool_support": 5,
        "rationale": "Small Llama-style model; RMSNorm and no bias match, but RoPE, SwiGLU and GQA require RTL changes.",
    },
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T": {
        "rtl_compatibility": 9,
        "simulation_scale": 5,
        "availability": 10,
        "architecture_clarity": 9,
        "license_clarity": 10,
        "kv_eviction_value": 10,
        "veda_closeness": 10,
        "tool_support": 5,
        "rationale": "Closest small public Llama-like baseline to VEDA, but large for RTL and uses GQA/SwiGLU/RoPE.",
    },
    "Qwen/Qwen2.5-0.5B": {
        "rtl_compatibility": 8,
        "simulation_scale": 11,
        "availability": 10,
        "architecture_clarity": 10,
        "license_clarity": 10,
        "kv_eviction_value": 10,
        "veda_closeness": 9,
        "tool_support": 5,
        "rationale": "Strong small eviction-research candidate; Qwen2 uses GQA, RoPE, SwiGLU and attention bias.",
    },
    "Qwen/Qwen3-0.6B": {
        "rtl_compatibility": 10,
        "simulation_scale": 10,
        "availability": 10,
        "architecture_clarity": 10,
        "license_clarity": 10,
        "kv_eviction_value": 10,
        "veda_closeness": 9,
        "tool_support": 5,
        "rationale": "Newer Qwen small model; no attention bias in config and useful GQA ratio, but still needs RoPE/SwiGLU/GQA.",
    },
    "EleutherAI/pythia-70m-deduped": {
        "rtl_compatibility": 8,
        "simulation_scale": 17,
        "availability": 10,
        "architecture_clarity": 8,
        "license_clarity": 10,
        "kv_eviction_value": 5,
        "veda_closeness": 4,
        "tool_support": 5,
        "rationale": "Small and open; GPT-NeoX parallel residual, GELU and partial RoPE differ from RTL.",
    },
    "distilgpt2": {
        "rtl_compatibility": 7,
        "simulation_scale": 13,
        "availability": 10,
        "architecture_clarity": 9,
        "license_clarity": 10,
        "kv_eviction_value": 4,
        "veda_closeness": 3,
        "tool_support": 5,
        "rationale": "Very well supported but old GPT-2 LayerNorm/GELU/bias/learned-position stack.",
    },
    "meta-llama/Llama-2-7b-hf": {
        "rtl_compatibility": 9,
        "simulation_scale": 0,
        "availability": 2,
        "architecture_clarity": 9,
        "license_clarity": 3,
        "kv_eviction_value": 10,
        "veda_closeness": 10,
        "tool_support": 5,
        "rationale": "VEDA reference family, but gated and too large for complete RTL simulation.",
    },
}


def total(score: dict[str, int | str]) -> int:
    return sum(int(score[key]) for key in WEIGHTS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", help="Optional CSV output path.")
    args = parser.parse_args()

    rows = []
    for model_id, score in SCORES.items():
        row = {"model_id": model_id, **score, "total": total(score)}
        rows.append(row)
    rows.sort(key=lambda item: int(item["total"]), reverse=True)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            fieldnames = ["model_id", *WEIGHTS, "total", "rationale"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    for row in rows:
        print(f"{row['total']:3d}  {row['model_id']}  {row['rationale']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
