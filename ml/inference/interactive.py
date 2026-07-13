"""Interactive and diagnostic inference helpers for ML-M2."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ml.architecture.causal_lm import HardwareMatchedCausalLM
from ml.architecture.config import HardwareMatchedConfig
from ml.data.dataset_hash import sha256_file
from ml.export.formal_export import _torch_load
from ml.inference.incremental_decode import compare_full_vs_incremental
from ml.tokenizer.load_tokenizer import SimpleBPETokenizer


DEFAULT_ARTIFACT_ROOT = Path("D:/IC_Workspace/VEDA_artifacts/ml_m2/formal")


@dataclass
class GenerationConfig:
    mode: str = "sample"
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    max_new_tokens: int = 64
    seed: int = 20260713
    filter_special_tokens: bool = True

    def validate(self) -> None:
        if self.mode not in {"greedy", "sample"}:
            raise ValueError("mode must be greedy or sample")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be >= 1.0")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")


@dataclass
class ModelBundle:
    artifact_root: Path
    model: HardwareMatchedCausalLM
    tokenizer: SimpleBPETokenizer
    checkpoint_path: Path
    checkpoint_sha256: str
    data_manifest: dict[str, Any]
    training_metrics: dict[str, Any]
    dataset_metadata: dict[str, Any]
    generation_config_path: Path
    generation_config_manifest: dict[str, Any]


def _ids_to_tokens(tokenizer: SimpleBPETokenizer, ids: list[int]) -> list[str]:
    return [tokenizer.id_to_token.get(int(token_id), "<missing>") for token_id in ids]


def load_interactive_bundle(artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT) -> ModelBundle:
    root = Path(artifact_root)
    data_manifest_path = root / "data" / "formal_data_manifest.json"
    metrics_path = root / "training" / "formal_training_metrics.json"
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(metrics["best_checkpoint"])
    tokenizer_path = Path(data_manifest["tokenizer"]["tokenizer_json"])
    generation_config_path = root / "interactive_traces" / "generation_config.json"
    if generation_config_path.exists():
        generation_config_manifest = json.loads(generation_config_path.read_text(encoding="utf-8"))
    else:
        generation_config_path.parent.mkdir(parents=True, exist_ok=True)
        generation_config_manifest = {
            "stage": "ML-M2 Post-Acceptance Interactive Evaluation",
            "default_interactive": asdict(GenerationConfig()),
            "hardware_compare": asdict(
                GenerationConfig(
                    mode="greedy",
                    temperature=1.0,
                    top_k=0,
                    top_p=1.0,
                    repetition_penalty=1.0,
                    max_new_tokens=64,
                )
            ),
        }
        generation_config_path.write_text(
            json.dumps(generation_config_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    payload = _torch_load(checkpoint_path)
    config = HardwareMatchedConfig.from_json_dict(payload["config"])
    model = HardwareMatchedCausalLM(config)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    tokenizer = SimpleBPETokenizer.load(tokenizer_path)
    return ModelBundle(
        artifact_root=root,
        model=model,
        tokenizer=tokenizer,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=sha256_file(checkpoint_path),
        data_manifest=data_manifest,
        training_metrics=metrics,
        dataset_metadata=data_manifest["dataset"],
        generation_config_path=generation_config_path,
        generation_config_manifest=generation_config_manifest,
    )


def apply_repetition_penalty(logits: torch.Tensor, token_ids: list[int], penalty: float) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    adjusted = logits.clone()
    for token_id in set(int(token) for token in token_ids):
        value = adjusted[token_id]
        adjusted[token_id] = value / penalty if value > 0 else value * penalty
    return adjusted


def filter_special_token_logits(logits: torch.Tensor, tokenizer: SimpleBPETokenizer, allow_eos: bool = True) -> torch.Tensor:
    filtered = logits.clone()
    blocked = [tokenizer.pad_id, tokenizer.bos_id, tokenizer.unk_id]
    if not allow_eos:
        blocked.append(tokenizer.eos_id)
    for token_id in blocked:
        filtered[int(token_id)] = -torch.inf
    return filtered


def filter_logits_top_k_top_p(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    filtered = logits.clone()
    if top_k > 0 and top_k < filtered.numel():
        threshold = torch.topk(filtered, top_k).values[-1]
        filtered = filtered.masked_fill(filtered < threshold, -torch.inf)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        if remove.numel():
            remove[1:] = remove[:-1].clone()
            remove[0] = False
        filtered[sorted_indices[remove]] = -torch.inf
    return filtered


def next_token_distribution(
    logits: torch.Tensor,
    tokenizer: SimpleBPETokenizer,
    k: int = 10,
    generated_ids: list[int] | None = None,
    config: GenerationConfig | None = None,
) -> dict[str, Any]:
    generated_ids = generated_ids or []
    config = config or GenerationConfig(mode="greedy", temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0)
    adjusted = apply_repetition_penalty(logits[-1].float(), generated_ids, config.repetition_penalty)
    if config.filter_special_tokens:
        adjusted = filter_special_token_logits(adjusted, tokenizer, allow_eos=True)
    if config.mode == "sample":
        adjusted = adjusted / config.temperature
        adjusted = filter_logits_top_k_top_p(adjusted, config.top_k, config.top_p)
    probs = torch.softmax(adjusted, dim=-1)
    entropy = float(-(probs * torch.log(probs.clamp_min(1e-30))).sum().item())
    top_values, top_indices = torch.topk(probs, k=min(k, probs.numel()))
    rows = []
    for probability, token_id in zip(top_values.tolist(), top_indices.tolist()):
        rows.append(
            {
                "token_id": int(token_id),
                "token": tokenizer.id_to_token.get(int(token_id), "<missing>"),
                "logit": float(logits[-1, int(token_id)].item()),
                "probability": float(probability),
            }
        )
    special_ids = {tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id, tokenizer.unk_id}
    return {
        "top": rows,
        "entropy": entropy,
        "special_token_probability": float(sum(float(probs[idx].item()) for idx in special_ids)),
        "eos_probability": float(probs[tokenizer.eos_id].item()),
    }


@torch.no_grad()
def generate_ids(
    model: HardwareMatchedCausalLM,
    tokenizer: SimpleBPETokenizer,
    prompt_ids: list[int],
    config: GenerationConfig,
) -> dict[str, Any]:
    config.validate()
    generator = torch.Generator(device="cpu").manual_seed(int(config.seed))
    generated = list(prompt_ids)
    new_ids: list[int] = []
    entropies: list[float] = []
    cache = None
    start = time.perf_counter()
    max_steps = min(config.max_new_tokens, max(0, model.config.context_length - len(prompt_ids)))
    for step in range(config.max_new_tokens):
        if step >= max_steps:
            break
        if step == 0:
            input_ids = torch.tensor([generated], dtype=torch.long)
            start_pos = 0
        else:
            input_ids = torch.tensor([[generated[-1]]], dtype=torch.long)
            start_pos = len(generated) - 1
        out = model(input_ids, past_kv=cache, use_cache=True, start_pos=start_pos)
        cache = out["past_kv"]
        logits = out["logits"][0, -1].float()
        adjusted = apply_repetition_penalty(logits, generated, config.repetition_penalty)
        if config.filter_special_tokens:
            adjusted = filter_special_token_logits(adjusted, tokenizer, allow_eos=True)
        if config.mode == "greedy":
            next_id = int(torch.argmax(adjusted).item())
            probs = torch.softmax(adjusted, dim=-1)
        else:
            adjusted = adjusted / config.temperature
            adjusted = filter_logits_top_k_top_p(adjusted, config.top_k, config.top_p)
            probs = torch.softmax(adjusted, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
        entropies.append(float(-(probs * torch.log(probs.clamp_min(1e-30))).sum().item()))
        generated.append(next_id)
        new_ids.append(next_id)
        if next_id == tokenizer.eos_id:
            break
    elapsed = time.perf_counter() - start
    return {
        "prompt_ids": prompt_ids,
        "generated_ids": new_ids,
        "all_ids": generated,
        "kv_cache_length": int(cache[0].valid_seq_len) if cache else len(prompt_ids),
        "generation_time_seconds": elapsed,
        "tokens_per_second": len(new_ids) / elapsed if elapsed > 0 else 0.0,
        "entropies": entropies,
        "hit_eos": bool(new_ids and new_ids[-1] == tokenizer.eos_id),
    }


def generate_text_record(bundle: ModelBundle, prompt: str, config: GenerationConfig) -> dict[str, Any]:
    prompt_ids = bundle.tokenizer.encode(prompt, add_bos=True)
    result = generate_ids(bundle.model, bundle.tokenizer, prompt_ids, config)
    generated_tokens = _ids_to_tokens(bundle.tokenizer, result["generated_ids"])
    prompt_tokens = _ids_to_tokens(bundle.tokenizer, prompt_ids)
    return {
        "prompt_text": prompt,
        "prompt_token_ids": prompt_ids,
        "prompt_tokens": prompt_tokens,
        "prompt_length": len(prompt_ids),
        "generated_token_ids": result["generated_ids"],
        "generated_tokens": generated_tokens,
        "generated_text": bundle.tokenizer.decode(result["all_ids"]),
        "kv_cache_length": result["kv_cache_length"],
        "generation_time_seconds": result["generation_time_seconds"],
        "tokens_per_second": result["tokens_per_second"],
        "hit_eos": result["hit_eos"],
        "average_entropy": sum(result["entropies"]) / len(result["entropies"]) if result["entropies"] else 0.0,
        "config": asdict(config),
    }


@torch.no_grad()
def next_token_report(bundle: ModelBundle, prompt: str, top_k: int = 10) -> dict[str, Any]:
    prompt_ids = bundle.tokenizer.encode(prompt, add_bos=True)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    out = bundle.model(input_ids)
    logits = out["logits"][0]
    dist = next_token_distribution(logits, bundle.tokenizer, k=top_k)
    return {
        "prompt_text": prompt,
        "tokenized_prompt": {
            "token_ids": prompt_ids,
            "tokens": _ids_to_tokens(bundle.tokenizer, prompt_ids),
        },
        "last_position": len(prompt_ids) - 1,
        "top_1": dist["top"][:1],
        "top_5": dist["top"][:5],
        "top_10": dist["top"][:10],
        "entropy": dist["entropy"],
        "special_token_probability": dist["special_token_probability"],
        "eos_probability": dist["eos_probability"],
    }


def _checksum_tensor(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def summarize_tensor(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    data = tensor.detach().float().cpu()
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": float(data.min().item()) if data.numel() else 0.0,
        "max": float(data.max().item()) if data.numel() else 0.0,
        "mean": float(data.mean().item()) if data.numel() else 0.0,
        "rms": float(torch.sqrt(torch.mean(data * data)).item()) if data.numel() else 0.0,
        "checksum": _checksum_tensor(tensor),
    }


@torch.no_grad()
def inspect_prompt(
    bundle: ModelBundle,
    prompt: str,
    output_dir: str | Path | None = None,
    full_tensor: bool = False,
    full_tensor_limit: int = 128,
) -> dict[str, Any]:
    prompt_ids = bundle.tokenizer.encode(prompt, add_bos=True)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    out = bundle.model(input_ids, use_cache=True, return_trace=True)
    trace = out["trace"]
    layer = trace["layer_0"]
    attn = layer["attention"]
    cache = out["past_kv"][0]
    tensors = {
        "token_ids": input_ids,
        "position_ids": trace["position_ids"],
        "token_embedding": trace["token_embedding"],
        "position_embedding": trace["position_embedding"],
        "layer_input": trace["layer_input"],
        "norm1": layer["rmsnorm1_output"],
        "Q": attn["q"],
        "K": attn["k"],
        "V": attn["v"],
        "attention_scores": attn["scores"],
        "attention_probabilities": attn["probabilities"],
        "attention_output": attn["head_output"],
        "residual1": layer["residual1"],
        "norm2": layer["rmsnorm2"],
        "W1": layer["w1_output"],
        "ReLU": layer["relu_output"],
        "W2": layer["w2_output"],
        "layer_output": trace["layer_output"],
        "final_norm": trace["final_norm"],
        "logits": trace["logits"],
        "top_k_values": trace["top_k"]["values"],
        "top_k_indices": trace["top_k"]["indices"],
        "K_cache": cache.k,
        "V_cache": cache.v,
    }
    records = []
    for name, tensor in tensors.items():
        record = summarize_tensor(name, tensor)
        if full_tensor and tensor.numel() <= full_tensor_limit:
            record["values"] = tensor.detach().cpu().tolist()
        records.append(record)
    report = {
        "prompt_text": prompt,
        "prompt_token_ids": prompt_ids,
        "prompt_tokens": _ids_to_tokens(bundle.tokenizer, prompt_ids),
        "checkpoint": str(bundle.checkpoint_path),
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "records": records,
    }
    target_dir = Path(output_dir) if output_dir else bundle.artifact_root / "interactive_traces"
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
    target = target_dir / f"inspect_{safe_name}.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["output_path"] = str(target)
    return report


def repetition_metrics(
    token_ids: list[int],
    special_ids: set[int] | None = None,
    token_texts: list[str] | None = None,
) -> dict[str, Any]:
    special_ids = special_ids or set()
    generated = [int(token) for token in token_ids]
    if not generated:
        return {
            "repeat_rate": 0.0,
            "distinct_1": 0.0,
            "distinct_2": 0.0,
            "ngram_loop_count": 0,
            "single_token_collapse": False,
            "special_token_ratio": 0.0,
        }
    repeat_edges = sum(1 for left, right in zip(generated, generated[1:]) if left == right)
    bigrams = list(zip(generated, generated[1:]))
    trigrams = list(zip(generated, generated[1:], generated[2:]))
    ngram_loop_count = (len(bigrams) - len(set(bigrams))) + (len(trigrams) - len(set(trigrams)))
    collapse_population = generated
    if token_texts is not None:
        collapse_population = [
            token
            for token, text in zip(generated, token_texts)
            if token not in special_ids and str(text).strip() != ""
        ]
    max_freq = max((collapse_population.count(token) for token in set(collapse_population)), default=0)
    special_count = sum(1 for token in generated if token in special_ids)
    return {
        "repeat_rate": repeat_edges / max(len(generated) - 1, 1),
        "distinct_1": len(set(generated)) / len(generated),
        "distinct_2": len(set(bigrams)) / len(bigrams) if bigrams else 0.0,
        "ngram_loop_count": ngram_loop_count,
        "single_token_collapse": bool(
            len(collapse_population) >= 8 and max_freq / max(len(collapse_population), 1) >= 0.5
        ),
        "special_token_ratio": special_count / len(generated),
    }


def config_variants() -> list[tuple[str, GenerationConfig]]:
    return [
        ("greedy", GenerationConfig(mode="greedy", temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0, max_new_tokens=48)),
        ("temperature_0_7", GenerationConfig(mode="sample", temperature=0.7, top_k=0, top_p=1.0, repetition_penalty=1.0, max_new_tokens=48)),
        ("temperature_0_8", GenerationConfig(mode="sample", temperature=0.8, top_k=0, top_p=1.0, repetition_penalty=1.0, max_new_tokens=48)),
        ("temperature_1_0", GenerationConfig(mode="sample", temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0, max_new_tokens=48)),
        ("top_k_20", GenerationConfig(mode="sample", temperature=0.8, top_k=20, top_p=1.0, repetition_penalty=1.0, max_new_tokens=48)),
        ("top_k_40", GenerationConfig(mode="sample", temperature=0.8, top_k=40, top_p=1.0, repetition_penalty=1.0, max_new_tokens=48)),
        ("top_p_0_9", GenerationConfig(mode="sample", temperature=0.8, top_k=0, top_p=0.9, repetition_penalty=1.0, max_new_tokens=48)),
        ("sample_default_penalty", GenerationConfig(max_new_tokens=48)),
    ]


def evaluate_prompt_suite(bundle: ModelBundle, prompts: list[str], output_path: str | Path | None = None) -> dict[str, Any]:
    special_ids = {bundle.tokenizer.pad_id, bundle.tokenizer.bos_id, bundle.tokenizer.eos_id, bundle.tokenizer.unk_id}
    results = []
    for prompt in prompts:
        for name, cfg in config_variants():
            record = generate_text_record(bundle, prompt, cfg)
            metrics = repetition_metrics(record["generated_token_ids"], special_ids, record["generated_tokens"])
            record.update({"variant": name, "metrics": metrics})
            results.append(record)
    by_variant = {}
    for name, _ in config_variants():
        rows = [row for row in results if row["variant"] == name]
        by_variant[name] = {
            "average_generated_length": sum(len(row["generated_token_ids"]) for row in rows) / max(len(rows), 1),
            "average_entropy": sum(row["average_entropy"] for row in rows) / max(len(rows), 1),
            "average_repeat_rate": sum(row["metrics"]["repeat_rate"] for row in rows) / max(len(rows), 1),
            "average_distinct_1": sum(row["metrics"]["distinct_1"] for row in rows) / max(len(rows), 1),
            "average_distinct_2": sum(row["metrics"]["distinct_2"] for row in rows) / max(len(rows), 1),
            "eos_rate": sum(1 for row in rows if row["hit_eos"]) / max(len(rows), 1),
            "average_special_token_ratio": sum(row["metrics"]["special_token_ratio"] for row in rows) / max(len(rows), 1),
            "single_token_collapse_count": sum(1 for row in rows if row["metrics"]["single_token_collapse"]),
            "ngram_loop_count": sum(row["metrics"]["ngram_loop_count"] for row in rows),
        }
    kv_checks = []
    for prompt in prompts:
        ids = torch.tensor([bundle.tokenizer.encode(prompt, add_bos=True)], dtype=torch.long)
        kv_checks.append({"prompt": prompt, **compare_full_vs_incremental(bundle.model, ids)})
    report = {
        "stage": "ML-M2 Post-Acceptance Interactive Evaluation",
        "checkpoint": str(bundle.checkpoint_path),
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "tokenizer": bundle.data_manifest["tokenizer"]["tokenizer_json"],
        "dataset": bundle.dataset_metadata,
        "prompt_count": len(prompts),
        "variant_summary": by_variant,
        "results": results,
        "incremental_kv_checks": kv_checks,
    }
    target = Path(output_path) if output_path else bundle.artifact_root / "interactive_traces" / "prompt_suite_results.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["output_path"] = str(target)
    return report


class ChatSession:
    def __init__(self, bundle: ModelBundle, config: GenerationConfig | None = None):
        self.bundle = bundle
        self.config = config or GenerationConfig()
        self.history: list[dict[str, Any]] = []

    def clear(self) -> None:
        self.history.clear()

    def set_seed(self, seed: int) -> None:
        self.config.seed = int(seed)

    def generate(self, prompt: str) -> dict[str, Any]:
        record = generate_text_record(self.bundle, prompt, self.config)
        self.history.append(record)
        return record
