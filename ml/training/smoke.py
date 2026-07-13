"""CPU smoke training flow for ML-M2."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from ml.architecture.config import HardwareMatchedConfig
from ml.architecture.causal_lm import HardwareMatchedCausalLM
from ml.data.dataset_hash import sha256_text
from ml.data.fixtures import SMOKE_STORIES, SMOKE_TEST_PROMPTS, fixture_text
from ml.data.sequence_builder import build_lm_sequences, deterministic_split
from ml.inference.generate import generate_text
from ml.inference.incremental_decode import compare_full_vs_incremental
from ml.tokenizer.train_bpe import train_tokenizer_from_texts
from ml.training.checkpoint import load_checkpoint, save_checkpoint
from ml.training.trainer import train_for_steps


def _config_from_stage_config(stage_config: dict) -> tuple[HardwareMatchedConfig, dict]:
    model_cfg = dict(stage_config["model"])
    seed = int(stage_config.get("training", {}).get("seed", 20260713))
    model_cfg["seed"] = seed
    return HardwareMatchedConfig(**model_cfg), stage_config["training"]


def run_smoke_training(output_dir: str | Path, stage_config: dict) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config, training_cfg = _config_from_stage_config(stage_config)
    train_docs, val_docs, test_prompts = deterministic_split(SMOKE_STORIES, validation_fraction=0.25, test_count=2)
    tokenizer, tok_manifest = train_tokenizer_from_texts(
        train_docs,
        out / "tokenizer",
        vocab_size=config.vocab_size,
        seed=config.seed,
        source_dataset="builtin_fixture",
        source_sha256=sha256_text(fixture_text()),
    )
    train_ids = []
    for doc in train_docs:
        train_ids.extend(tokenizer.encode(doc, add_bos=True, add_eos=True))
    val_ids = []
    for doc in val_docs:
        val_ids.extend(tokenizer.encode(doc, add_bos=True, add_eos=True))
    train_batch = build_lm_sequences(train_ids, config.context_length, tokenizer.pad_id, stride=max(1, config.context_length // 2))
    val_batch = build_lm_sequences(val_ids, config.context_length, tokenizer.pad_id, stride=max(1, config.context_length // 2))
    model, metrics = train_for_steps(
        config=config,
        train_batch=train_batch,
        validation_batch=val_batch,
        steps=int(training_cfg.get("steps", 12)),
        batch_size=int(training_cfg.get("batch_size", 8)),
        learning_rate=float(training_cfg.get("learning_rate", 0.003)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
        grad_clip_norm=float(training_cfg.get("grad_clip_norm", 1.0)),
        seed=config.seed,
        device="cpu",
    )
    ckpt_path = out / "checkpoints" / "ml_m2_smoke_last.pt"
    checkpoint_manifest = save_checkpoint(
        ckpt_path,
        model,
        optimizer=None,
        step=metrics.steps,
        config=config.to_json_dict(),
        metrics=asdict(metrics),
    )
    reload_model = HardwareMatchedCausalLM(config)
    load_checkpoint(ckpt_path, reload_model)
    sample_ids = torch.tensor([tokenizer.encode(SMOKE_TEST_PROMPTS[0], add_bos=True)], dtype=torch.long)
    with torch.no_grad():
        model_logits = model(sample_ids)["logits"]
        reload_logits = reload_model(sample_ids)["logits"]
    reload_max_abs_diff = float((model_logits - reload_logits).abs().max().item())
    generation = generate_text(model, tokenizer, SMOKE_TEST_PROMPTS[0], max_new_tokens=8)
    compare = compare_full_vs_incremental(model, sample_ids)
    generation_path = out / "generation_samples.json"
    generation_path.write_text(
        json.dumps({"prompt": SMOKE_TEST_PROMPTS[0], "generated": generation, "test_prompts": test_prompts}, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "metrics": asdict(metrics),
        "checkpoint_manifest": checkpoint_manifest,
        "tokenizer_manifest": asdict(tok_manifest),
        "reload_max_abs_diff": reload_max_abs_diff,
        "generation_path": str(generation_path),
        "incremental_compare": compare,
    }
    (out / "smoke_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "model": model,
        "tokenizer": tokenizer,
        "metrics": metrics,
        "checkpoint_manifest": checkpoint_manifest,
        "reload_max_abs_diff": reload_max_abs_diff,
        "generation_path": generation_path,
        "incremental_compare": compare,
    }
