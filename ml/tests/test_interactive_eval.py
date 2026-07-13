import json
from pathlib import Path

import torch

from ml.architecture.causal_lm import HardwareMatchedCausalLM
from ml.architecture.config import HardwareMatchedConfig
from ml.inference.interactive import (
    ChatSession,
    GenerationConfig,
    apply_repetition_penalty,
    evaluate_prompt_suite,
    filter_logits_top_k_top_p,
    generate_text_record,
    inspect_prompt,
    load_interactive_bundle,
)
from ml.tokenizer.load_tokenizer import SimpleBPETokenizer


def make_artifact_root(tmp_path: Path):
    torch.manual_seed(123)
    texts = ["Once upon a time there was a small dog.", "The red ball was in the garden."]
    tokenizer = SimpleBPETokenizer.train(texts, vocab_size=64)
    root = tmp_path / "formal"
    tokenizer_dir = root / "tokenizer"
    tokenizer_dir.mkdir(parents=True)
    tokenizer.save(tokenizer_dir / "tokenizer.json")
    cfg = HardwareMatchedConfig(vocab_size=len(tokenizer.vocab), context_length=32)
    model = HardwareMatchedCausalLM(cfg).eval()
    ckpt = root / "checkpoints" / "best.pt"
    ckpt.parent.mkdir(parents=True)
    torch.save({"config": cfg.to_json_dict(), "model_state_dict": model.state_dict()}, ckpt)
    data_manifest = {
        "dataset": {"name": "fixture", "license": "test"},
        "tokenizer": {"tokenizer_json": str(tokenizer_dir / "tokenizer.json")},
        "packing": {"context_length": cfg.context_length},
    }
    metrics = {"best_checkpoint": str(ckpt), "best_validation_loss": 1.0}
    (root / "data").mkdir()
    (root / "training").mkdir()
    (root / "data" / "formal_data_manifest.json").write_text(json.dumps(data_manifest), encoding="utf-8")
    (root / "training" / "formal_training_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return root


def test_checkpoint_and_tokenizer_load(tmp_path: Path):
    bundle = load_interactive_bundle(make_artifact_root(tmp_path))
    assert bundle.model.config.context_length == 32
    assert bundle.tokenizer.bos_id == 1
    assert bundle.checkpoint_path.exists()


def test_deterministic_greedy_and_sampling_fixed_seed(tmp_path: Path):
    bundle = load_interactive_bundle(make_artifact_root(tmp_path))
    greedy = GenerationConfig(mode="greedy", max_new_tokens=4, repetition_penalty=1.0)
    a = generate_text_record(bundle, "Once upon", greedy)
    b = generate_text_record(bundle, "Once upon", greedy)
    assert a["generated_token_ids"] == b["generated_token_ids"]
    sample = GenerationConfig(mode="sample", max_new_tokens=4, seed=7, top_k=10, top_p=0.9)
    c = generate_text_record(bundle, "Once upon", sample)
    d = generate_text_record(bundle, "Once upon", sample)
    assert c["generated_token_ids"] == d["generated_token_ids"]


def test_top_k_top_p_and_repetition_penalty():
    logits = torch.tensor([0.0, 1.0, 2.0, 3.0])
    top_k = filter_logits_top_k_top_p(logits, top_k=2, top_p=1.0)
    assert torch.isneginf(top_k[0])
    assert torch.isneginf(top_k[1])
    assert torch.isfinite(top_k[2])
    top_p = filter_logits_top_k_top_p(logits, top_k=0, top_p=0.7)
    assert torch.isfinite(top_p).sum().item() < logits.numel()
    penalized = apply_repetition_penalty(torch.tensor([2.0, -2.0]), [0, 1], 2.0)
    assert penalized[0].item() == 1.0
    assert penalized[1].item() == -4.0


def test_special_token_filter_incremental_and_trace(tmp_path: Path):
    bundle = load_interactive_bundle(make_artifact_root(tmp_path))
    cfg = GenerationConfig(mode="greedy", max_new_tokens=3, filter_special_tokens=True)
    record = generate_text_record(bundle, "The red", cfg)
    blocked = {bundle.tokenizer.pad_id, bundle.tokenizer.bos_id, bundle.tokenizer.unk_id}
    assert not blocked.intersection(record["generated_token_ids"])
    ids = torch.tensor([bundle.tokenizer.encode("The red", add_bos=True)], dtype=torch.long)
    from ml.inference.incremental_decode import compare_full_vs_incremental

    compare = compare_full_vs_incremental(bundle.model, ids)
    assert compare["allclose"]
    trace = inspect_prompt(bundle, "The red", output_dir=tmp_path / "traces")
    assert Path(trace["output_path"]).exists()
    assert any(row["name"] == "Q" for row in trace["records"])


def test_chat_clear_reset_and_prompt_suite(tmp_path: Path):
    bundle = load_interactive_bundle(make_artifact_root(tmp_path))
    session = ChatSession(bundle, GenerationConfig(mode="greedy", max_new_tokens=2))
    session.generate("Once")
    assert session.history
    session.clear()
    assert session.history == []
    session.set_seed(99)
    assert session.config.seed == 99
    suite = evaluate_prompt_suite(bundle, ["Once upon a time"], output_path=tmp_path / "suite.json")
    assert Path(suite["output_path"]).exists()
    assert "greedy" in suite["variant_summary"]
