"""Formal checkpoint evaluation, FP16 export, and trace generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ml.architecture.causal_lm import HardwareMatchedCausalLM
from ml.architecture.config import HardwareMatchedConfig
from ml.cosim.fp16_policy import fp16_bits_nested_to_tensor
from ml.cosim.hardware_aware_model import run_hardware_aware_model
from ml.cosim.rtl_vector_builder import write_small_rtl_fixture
from ml.data.dataset_hash import sha256_file
from ml.data.dataset_manifest import artifact_root
from ml.evaluation.evaluate_quantization import logits_agreement, tensor_error_metrics
from ml.export.export_checkpoint import export_checkpoint
from ml.export.export_trace import export_trace
from ml.export.validate_export import validate_export_manifest
from ml.inference.generate import generate_text
from ml.tokenizer.load_tokenizer import SimpleBPETokenizer


def formal_root() -> Path:
    return artifact_root() / "formal"


def _torch_load(path: str | Path):
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location="cpu")


def _load_formal_manifest(root: Path) -> dict:
    return json.loads((root / "data" / "formal_data_manifest.json").read_text(encoding="utf-8"))


def load_formal_model(checkpoint_path: str | Path) -> tuple[HardwareMatchedCausalLM, dict]:
    payload = _torch_load(checkpoint_path)
    cfg = HardwareMatchedConfig.from_json_dict(payload["config"])
    model = HardwareMatchedCausalLM(cfg)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def _fp16_weight_rounded_model(model: HardwareMatchedCausalLM) -> HardwareMatchedCausalLM:
    clone = HardwareMatchedCausalLM(model.config)
    clone.load_state_dict(model.state_dict())
    with torch.no_grad():
        for param in clone.parameters():
            param.copy_(param.detach().half().float())
    clone.eval()
    return clone


def _ids_for_length(tokenizer: SimpleBPETokenizer, length: int) -> torch.Tensor:
    seed_text = "Once upon a time there was a small red bird who liked kind stories in the garden."
    ids = tokenizer.encode(seed_text, add_bos=True)
    filler = [idx for idx in ids if idx not in {tokenizer.pad_id, tokenizer.eos_id}]
    while len(filler) < length:
        filler.extend(filler[1:] or [tokenizer.bos_id])
    return torch.tensor([filler[:length]], dtype=torch.long)


@torch.no_grad()
def _compare_paths(model: HardwareMatchedCausalLM, input_ids: torch.Tensor) -> dict:
    fp16_model = _fp16_weight_rounded_model(model)
    pt = model(input_ids, return_trace=True)
    fp16 = fp16_model(input_ids)
    hw = run_hardware_aware_model(model, input_ids)
    pt_logits = pt["logits"]
    fp16_logits = fp16["logits"]
    hw_logits = hw["logits"]
    pt_k = pt["trace"]["layer_0"]["attention"]["k"][0].detach().float()
    pt_v = pt["trace"]["layer_0"]["attention"]["v"][0].detach().float()
    hw_k = fp16_bits_nested_to_tensor(hw["k_cache"])
    hw_v = fp16_bits_nested_to_tensor(hw["v_cache"])
    return {
        "pytorch_vs_fp16_weight": {
            **tensor_error_metrics(pt_logits, fp16_logits),
            **logits_agreement(pt_logits, fp16_logits),
        },
        "pytorch_vs_hardware_aware": {
            **tensor_error_metrics(pt_logits, hw_logits),
            **logits_agreement(pt_logits, hw_logits),
        },
        "layer_output_error": tensor_error_metrics(pt["trace"]["layer_output"], hw["layer_output"]),
        "kv_cache_error": {
            "k": tensor_error_metrics(pt_k, hw_k),
            "v": tensor_error_metrics(pt_v, hw_v),
        },
    }


def evaluate_formal_checkpoint(root: str | Path | None = None) -> dict:
    root_path = Path(root) if root else formal_root()
    metrics = json.loads((root_path / "training" / "formal_training_metrics.json").read_text(encoding="utf-8"))
    manifest = _load_formal_manifest(root_path)
    model, _ = load_formal_model(metrics["best_checkpoint"])
    tokenizer = SimpleBPETokenizer.load(manifest["tokenizer"]["tokenizer_json"])
    prompts = json.loads(Path(manifest["test_prompts"]["path"]).read_text(encoding="utf-8"))["prompts"]
    generations = []
    special_total = 0
    generated_total = 0
    first_token_distribution = []
    for prompt in prompts[:5]:
        encoded = torch.tensor([tokenizer.encode(prompt, add_bos=True)], dtype=torch.long)
        out = model(encoded)
        probs = torch.softmax(out["logits"][:, -1, :], dim=-1)
        top_values, top_indices = torch.topk(probs, k=5, dim=-1)
        generated_ids = model.generate_greedy(encoded, max_new_tokens=24, eos_token_id=tokenizer.eos_id)[0].tolist()
        new_ids = generated_ids[len(encoded[0]) :]
        special_total += sum(1 for token_id in new_ids if token_id in {tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id, tokenizer.unk_id})
        generated_total += len(new_ids)
        first_token_distribution.append(
            {
                "prompt": prompt,
                "top_ids": top_indices[0].tolist(),
                "top_probs": [float(value) for value in top_values[0].tolist()],
            }
        )
        generations.append({"prompt": prompt, "generated": tokenizer.decode(generated_ids)})
    reload_model, _ = load_formal_model(metrics["best_checkpoint"])
    sample = _ids_for_length(tokenizer, 8)
    reload_diff = float((model(sample)["logits"] - reload_model(sample)["logits"]).abs().max().item())
    result = {
        "stage": "ML-M2 Formal",
        "best_checkpoint": metrics["best_checkpoint"],
        "best_checkpoint_sha256": sha256_file(metrics["best_checkpoint"]),
        "untrained_validation_loss": metrics["untrained_validation_loss"],
        "best_validation_loss": metrics["best_validation_loss"],
        "validation_perplexity": metrics["perplexity"],
        "generation_samples": generations,
        "first_token_distribution": first_token_distribution,
        "special_token_output_ratio": special_total / max(generated_total, 1),
        "checkpoint_reload_max_abs_diff": reload_diff,
    }
    out_path = root_path / "evaluation" / "formal_evaluation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def export_formal_checkpoint(root: str | Path | None = None) -> dict:
    root_path = Path(root) if root else formal_root()
    metrics = json.loads((root_path / "training" / "formal_training_metrics.json").read_text(encoding="utf-8"))
    manifest = _load_formal_manifest(root_path)
    model, _ = load_formal_model(metrics["best_checkpoint"])
    tokenizer = SimpleBPETokenizer.load(manifest["tokenizer"]["tokenizer_json"])
    export_dir = root_path / "exports" / "fp16_best"
    export_manifest = export_checkpoint(metrics["best_checkpoint"], export_dir)
    validate_export_manifest(export_dir / "export_manifest.json")
    trace_dir = root_path / "traces"
    trace_results = []
    comparisons = {}
    for length in [1, 2, 8, 16]:
        input_ids = _ids_for_length(tokenizer, length)
        trace_manifest = export_trace(model, input_ids, trace_dir / f"trace_len_{length}.json")
        write_small_rtl_fixture(trace_dir / f"rtl_fixture_len_{length}.json", trace_manifest)
        trace_results.append(
            {
                "prompt_length": length,
                "trace_manifest": str(trace_dir / f"trace_len_{length}.json"),
                "trace_sha256": sha256_file(trace_dir / f"trace_len_{length}.json"),
                "rtl_fixture": str(trace_dir / f"rtl_fixture_len_{length}.json"),
                "rtl_fixture_sha256": sha256_file(trace_dir / f"rtl_fixture_len_{length}.json"),
                "trace_node_count": trace_manifest["trace_node_count"],
            }
        )
        comparisons[f"len_{length}"] = _compare_paths(model, input_ids)
    result = {
        "stage": "ML-M2 Formal",
        "best_checkpoint": metrics["best_checkpoint"],
        "best_checkpoint_sha256": sha256_file(metrics["best_checkpoint"]),
        "export_dir": str(export_dir),
        "export_manifest": str(export_dir / "export_manifest.json"),
        "export_manifest_sha256": sha256_file(export_dir / "export_manifest.json"),
        "export_tensor_count": export_manifest.tensor_count,
        "export_tensors": [record.logical_name for record in export_manifest.records],
        "trace_results": trace_results,
        "comparisons": comparisons,
    }
    out_path = root_path / "exports" / "formal_export_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(formal_root()))
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    payload = {}
    if not args.export_only:
        payload["evaluation"] = evaluate_formal_checkpoint(args.root)
    if not args.eval_only:
        payload["export"] = export_formal_checkpoint(args.root)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
