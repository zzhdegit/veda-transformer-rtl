import math
from pathlib import Path

import torch

from ml.architecture.causal_lm import HardwareMatchedCausalLM
from ml.architecture.config import HardwareMatchedConfig
from ml.data.sequence_builder import build_lm_sequences
from ml.training.numeric_audit import random_batch, single_batch_overfit, untrained_loss, validate_batch


def test_label_shift():
    batch = build_lm_sequences([1, 10, 11, 2], context_length=3, pad_id=0)
    assert batch.input_ids == [[1, 10, 11]]
    assert batch.labels == [[10, 11, 2]]


def test_padding_ignore():
    batch = build_lm_sequences([1, 10, 2], context_length=5, pad_id=0)
    assert batch.input_ids[0][-2:] == [0, 0]
    assert batch.labels[0][-3:] == [-100, -100, -100]


def test_vocab_range():
    input_ids = torch.tensor([[1, 4, 5, 2]])
    labels = torch.tensor([[4, 5, 2, -100]])
    validate_batch(input_ids, labels, vocab_size=8)


def test_logits_layout():
    cfg = HardwareMatchedConfig(vocab_size=32, context_length=8)
    model = HardwareMatchedCausalLM(cfg)
    ids, labels = random_batch(cfg.vocab_size, cfg.context_length, batch_size=2)
    out = model(ids, labels=labels)
    assert out["logits"].shape == (2, 8, 32)
    assert out["loss"].ndim == 0


def test_tied_lm_head():
    cfg = HardwareMatchedConfig(vocab_size=32, context_length=8, tie_word_embeddings=True)
    model = HardwareMatchedCausalLM(cfg)
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()


def test_initial_loss_scale():
    for vocab_size in (256, 2048):
        result = untrained_loss(vocab_size=vocab_size, context_length=16)
        assert result["passes_2x_log_vocab"], result
        assert result["loss"] < 2.0 * math.log(vocab_size)


def test_single_batch_overfit(tmp_path: Path):
    result = single_batch_overfit(steps=250, output_dir=tmp_path)
    assert result["final_loss"] < 1.0, result
    assert result["final_loss"] < result["initial_loss"]
    assert result["top1"] > 0.7
    assert result["reload_max_abs_diff"] == 0.0

