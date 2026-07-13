#!/usr/bin/env python3
"""Inspect public model metadata without downloading model weights.

The script uses Hugging Face model API and raw config.json endpoints only. It
does not pass tokens, does not execute model code, and never uses
trust_remote_code.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from typing import Any


CANDIDATES: dict[str, dict[str, Any]] = {
    "facebook/opt-125m": {
        "route": "A",
        "detail": True,
        "params_label": "125M",
        "notes": "Pre-LN OPT baseline with ReLU FFN; LayerNorm and bias differ from RTL.",
    },
    "roneneldan/TinyStories-1M": {
        "route": "A",
        "detail": True,
        "params_label": "1M advertised, 48.6 MB checkpoint",
        "notes": "Very small trained GPT-Neo/TinyStories checkpoint; local attention and GELU differ.",
    },
    "roneneldan/TinyStories-3M": {
        "route": "A",
        "detail": False,
        "params_label": "3M advertised",
        "notes": "TinyStories GPT-Neo family; larger than 1M, same structural mismatches.",
    },
    "roneneldan/TinyStories-8M": {
        "route": "A",
        "detail": False,
        "params_label": "8M advertised",
        "notes": "TinyStories GPT-Neo family; D_MODEL=256 exceeds current RMSNorm checked range.",
    },
    "roneneldan/TinyStories-33M": {
        "route": "A",
        "detail": False,
        "params_label": "33M advertised",
        "notes": "TinyStories GPT-Neo family; hidden size 768 makes RTL simulation heavier.",
    },
    "HuggingFaceTB/SmolLM-135M": {
        "route": "B",
        "detail": False,
        "params_label": "135M",
        "notes": "Earlier SmolLM small Llama-style checkpoint.",
    },
    "HuggingFaceTB/SmolLM2-135M": {
        "route": "B",
        "detail": True,
        "params_label": "135M",
        "notes": "Smallest high-quality Llama-style candidate with RMSNorm/RoPE/SwiGLU/GQA.",
    },
    "HuggingFaceTB/SmolLM2-360M": {
        "route": "B",
        "detail": False,
        "params_label": "360M",
        "notes": "Same family as SmolLM2-135M with larger hidden size.",
    },
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T": {
        "route": "B",
        "detail": True,
        "params_label": "1.1B",
        "notes": "Llama-style trained base checkpoint; close to VEDA paper family but larger.",
    },
    "Qwen/Qwen2.5-0.5B": {
        "route": "B",
        "detail": True,
        "params_label": "0.5B",
        "notes": "Small Qwen2.5 base model with GQA, RoPE, SwiGLU, QKV bias.",
    },
    "Qwen/Qwen3-0.6B": {
        "route": "B",
        "detail": True,
        "params_label": "0.6B",
        "notes": "Newer Qwen3 small model; GQA, RoPE, SwiGLU, no attention bias in config.",
    },
    "EleutherAI/pythia-70m-deduped": {
        "route": "A",
        "detail": False,
        "params_label": "70M",
        "notes": "Small GPT-NeoX baseline; GELU and partial RoPE differ.",
    },
    "EleutherAI/pythia-160m-deduped": {
        "route": "A",
        "detail": False,
        "params_label": "160M",
        "notes": "Larger Pythia baseline; same structural mismatches as 70M.",
    },
    "distilgpt2": {
        "route": "A",
        "detail": False,
        "params_label": "82M",
        "notes": "Very well supported GPT-2 family baseline; LayerNorm/GELU/bias.",
    },
    "meta-llama/Llama-2-7b-hf": {
        "route": "B",
        "detail": True,
        "params_label": "7B",
        "notes": "VEDA paper-family reference; gated weights and too large for full RTL.",
    },
}


def fetch_json(url: str) -> tuple[dict[str, Any] | None, str | None]:
    request = urllib.request.Request(url, headers={"User-Agent": "VEDA-model-selection/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response), None
    except urllib.error.HTTPError as exc:
        return None, f"HTTP {exc.code}: {exc.reason}"
    except Exception as exc:  # pragma: no cover - diagnostic path
        return None, f"{type(exc).__name__}: {exc}"


def model_api_url(model_id: str) -> str:
    return "https://huggingface.co/api/models/" + urllib.parse.quote(model_id, safe="/") + "?blobs=true"


def config_url(model_id: str) -> str:
    return f"https://huggingface.co/{model_id}/raw/main/config.json"


def derive_head_dim(config: dict[str, Any]) -> int | None:
    if config.get("head_dim") is not None:
        return int(config["head_dim"])
    hidden = config.get("hidden_size") or config.get("n_embd") or config.get("d_model")
    heads = config.get("num_attention_heads") or config.get("n_head") or config.get("num_heads")
    if hidden and heads:
        return int(hidden) // int(heads)
    return None


def derive_layers(config: dict[str, Any]) -> int | None:
    for key in ("num_hidden_layers", "n_layer", "num_layers"):
        if config.get(key) is not None:
            return int(config[key])
    return None


def derive_hidden(config: dict[str, Any]) -> int | None:
    for key in ("hidden_size", "n_embd", "d_model", "word_embed_proj_dim"):
        if config.get(key) is not None:
            return int(config[key])
    return None


def derive_heads(config: dict[str, Any]) -> int | None:
    for key in ("num_attention_heads", "n_head", "num_heads"):
        if config.get(key) is not None:
            return int(config[key])
    return None


def derive_kv_heads(config: dict[str, Any]) -> int | None:
    if config.get("num_key_value_heads") is not None:
        return int(config["num_key_value_heads"])
    return derive_heads(config)


def derive_intermediate(config: dict[str, Any]) -> int | None:
    for key in ("intermediate_size", "ffn_dim", "n_inner"):
        if config.get(key) is not None:
            return int(config[key])
    hidden = derive_hidden(config)
    model_type = config.get("model_type")
    if hidden and model_type in {"gpt2", "gpt_neo"}:
        return 4 * hidden
    return None


def derive_context(config: dict[str, Any]) -> int | None:
    for key in ("max_position_embeddings", "n_positions", "seq_length"):
        if config.get(key) is not None:
            return int(config[key])
    return None


def checkpoint_bytes(siblings: list[dict[str, Any]]) -> int:
    suffixes = (".bin", ".safetensors", ".pt", ".ckpt")
    return sum(
        int(item.get("size") or 0)
        for item in siblings
        if str(item.get("rfilename") or "").endswith(suffixes)
    )


def inspect_one(model_id: str) -> dict[str, Any]:
    now = datetime.now(UTC).date().isoformat()
    entry = dict(CANDIDATES.get(model_id, {}))
    api, api_error = fetch_json(model_api_url(model_id))
    config, config_error = fetch_json(config_url(model_id))
    card = (api or {}).get("cardData") or {}
    siblings = (api or {}).get("siblings") or []
    config = config or {}

    return {
        "query_date": now,
        "model_id": model_id,
        "route": entry.get("route", ""),
        "detail": bool(entry.get("detail", False)),
        "params_label": entry.get("params_label", ""),
        "revision": (api or {}).get("sha", ""),
        "last_modified": (api or {}).get("lastModified", ""),
        "gated": (api or {}).get("gated", ""),
        "license": card.get("license") or "",
        "library": (api or {}).get("library_name", ""),
        "pipeline": (api or {}).get("pipeline_tag", ""),
        "api_error": api_error or "",
        "config_error": config_error or "",
        "model_type": config.get("model_type", ""),
        "architectures": ",".join(config.get("architectures") or []),
        "layers": derive_layers(config) or "",
        "hidden_size": derive_hidden(config) or "",
        "num_attention_heads": derive_heads(config) or "",
        "num_key_value_heads": derive_kv_heads(config) or "",
        "head_dim": derive_head_dim(config) or "",
        "intermediate_size": derive_intermediate(config) or "",
        "hidden_act": config.get("hidden_act") or config.get("activation_function") or "",
        "norm_eps": config.get("rms_norm_eps") or config.get("layer_norm_epsilon") or config.get("layer_norm_eps") or "",
        "max_position_embeddings": derive_context(config) or "",
        "vocab_size": config.get("vocab_size", ""),
        "rope_theta": config.get("rope_theta") or config.get("rotary_emb_base") or "",
        "rope_scaling": json.dumps(config.get("rope_scaling"), sort_keys=True),
        "attention_bias": config.get("attention_bias", ""),
        "qkv_bias": config.get("qkv_bias", ""),
        "tie_word_embeddings": config.get("tie_word_embeddings", ""),
        "sliding_window": config.get("sliding_window", ""),
        "use_sliding_window": config.get("use_sliding_window", ""),
        "attention_types": json.dumps(config.get("attention_types") or config.get("attention_layers") or ""),
        "window_size": config.get("window_size", ""),
        "torch_dtype": config.get("torch_dtype", ""),
        "checkpoint_bytes": checkpoint_bytes(siblings),
        "safetensors": sum(1 for item in siblings if str(item.get("rfilename") or "").endswith(".safetensors")),
        "notes": entry.get("notes", ""),
        "config_url": config_url(model_id),
        "api_url": model_api_url(model_id),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=list(CANDIDATES), help="Model IDs to inspect.")
    parser.add_argument("--csv", help="Optional CSV output path.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text table.")
    args = parser.parse_args()

    rows = [inspect_one(model_id) for model_id in args.models]
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        for row in rows:
            print(
                f"{row['model_id']}: type={row['model_type']} layers={row['layers']} "
                f"D={row['hidden_size']} H={row['num_attention_heads']} "
                f"KV={row['num_key_value_heads']} act={row['hidden_act']} "
                f"license={row['license'] or 'NOT_DECLARED'} revision={row['revision'] or 'UNAVAILABLE'}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
