"""Numeric sanity checks before ML-M2 formal training."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ml.architecture.causal_lm import HardwareMatchedCausalLM
from ml.architecture.config import HardwareMatchedConfig
from ml.data.sequence_builder import SequenceBatch
from ml.training.checkpoint import load_checkpoint, save_checkpoint
from ml.training.optimizer import build_optimizer
from ml.training.reproducibility import set_seed


@dataclass(frozen=True)
class InitStats:
    tensor: str
    mean: float
    std: float
    max_abs: float


def tensor_stats(name: str, tensor: torch.Tensor) -> InitStats:
    values = tensor.detach().float()
    return InitStats(
        tensor=name,
        mean=float(values.mean().item()),
        std=float(values.std(unbiased=False).item()),
        max_abs=float(values.abs().max().item()),
    )


def initialization_report(model: HardwareMatchedCausalLM, input_ids: torch.Tensor) -> dict:
    with torch.no_grad():
        out = model(input_ids, return_trace=True)
        logits = out["logits"]
        hidden = out["trace"]["layer_input"]
    stats = [
        tensor_stats("token_embedding.weight", model.token_embedding.weight),
        tensor_stats("position_embedding.weight", model.position_embedding.weight),
        tensor_stats("layers.0.attn.wq.weight", model.layers[0].attn.wq.weight),
        tensor_stats("layers.0.attn.wk.weight", model.layers[0].attn.wk.weight),
        tensor_stats("layers.0.attn.wv.weight", model.layers[0].attn.wv.weight),
        tensor_stats("layers.0.attn.wo.weight", model.layers[0].attn.wo.weight),
        tensor_stats("layers.0.ffn.w1.weight", model.layers[0].ffn.w1.weight),
        tensor_stats("layers.0.ffn.w2.weight", model.layers[0].ffn.w2.weight),
        tensor_stats("final_norm.weight", model.final_norm.weight),
        tensor_stats("layer_input", hidden),
        tensor_stats("logits", logits),
    ]
    return {"stats": [asdict(row) for row in stats], "logits_shape": list(logits.shape)}


def random_batch(vocab_size: int, context_length: int, batch_size: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(4, vocab_size, (batch_size, context_length), dtype=torch.long)
    input_ids[:, 0] = 1
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    labels[:, -1] = 2
    return input_ids, labels


@torch.no_grad()
def untrained_loss(vocab_size: int, context_length: int = 32, seed: int = 20260713) -> dict:
    set_seed(seed)
    cfg = HardwareMatchedConfig(vocab_size=vocab_size, context_length=context_length)
    model = HardwareMatchedCausalLM(cfg).eval()
    input_ids, labels = random_batch(vocab_size, context_length)
    out = model(input_ids, labels=labels)
    loss = float(out["loss"].item())
    baseline = math.log(vocab_size)
    return {
        "vocab_size": vocab_size,
        "loss": loss,
        "log_vocab": baseline,
        "ratio_to_log_vocab": loss / baseline,
        "passes_2x_log_vocab": loss <= 2.0 * baseline,
        "initialization": initialization_report(model, input_ids),
    }


def validate_batch(input_ids: torch.Tensor, labels: torch.Tensor, vocab_size: int) -> None:
    if input_ids.ndim != 2 or labels.shape != input_ids.shape:
        raise ValueError("input_ids and labels must be [batch, sequence]")
    if int(input_ids.min().item()) < 0 or int(input_ids.max().item()) >= vocab_size:
        raise ValueError("input token out of vocabulary range")
    active = labels != -100
    if bool(active.any()):
        active_labels = labels[active]
        if int(active_labels.min().item()) < 0 or int(active_labels.max().item()) >= vocab_size:
            raise ValueError("target token out of vocabulary range")


def single_batch_overfit(
    vocab_size: int = 64,
    context_length: int = 16,
    steps: int = 250,
    learning_rate: float = 0.03,
    seed: int = 20260713,
    output_dir: str | Path | None = None,
) -> dict:
    set_seed(seed)
    cfg = HardwareMatchedConfig(vocab_size=vocab_size, context_length=context_length)
    model = HardwareMatchedCausalLM(cfg)
    input_ids, labels = random_batch(vocab_size, context_length, batch_size=4)
    validate_batch(input_ids, labels, vocab_size)
    optimizer = build_optimizer(model, learning_rate=learning_rate, weight_decay=0.0)
    losses: list[float] = []
    start = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids, labels=labels)
        loss = out["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError("single-batch overfit produced NaN/Inf")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().item()))
        if losses[-1] < 0.5:
            break
    elapsed = time.perf_counter() - start
    with torch.no_grad():
        logits = model(input_ids)["logits"]
        predictions = torch.argmax(logits, dim=-1)
        active = labels != -100
        top1 = float((predictions[active] == labels[active]).float().mean().item())
    reload_max_abs_diff = None
    ckpt_manifest = None
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt = out_dir / "single_batch_overfit.pt"
        ckpt_manifest = save_checkpoint(
            ckpt,
            model,
            optimizer,
            step=len(losses),
            config=cfg.to_json_dict(),
            metrics={"final_loss": losses[-1], "top1": top1},
        )
        reloaded = HardwareMatchedCausalLM(cfg)
        load_checkpoint(ckpt, reloaded)
        with torch.no_grad():
            reload_logits = reloaded(input_ids)["logits"]
        reload_max_abs_diff = float((logits - reload_logits).abs().max().item())
    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "steps": len(losses),
        "top1": top1,
        "elapsed_seconds": elapsed,
        "checkpoint_manifest": ckpt_manifest,
        "reload_max_abs_diff": reload_max_abs_diff,
    }


def write_numeric_audit(path: str | Path, artifact_dir: str | Path) -> dict:
    report = {
        "untrained_vocab_256": untrained_loss(256),
        "untrained_vocab_2048": untrained_loss(2048),
        "single_batch_overfit": single_batch_overfit(output_dir=artifact_dir),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    report = write_numeric_audit(
        "reports/ml_m2/pretraining_numeric_audit.json",
        "D:/IC_Workspace/VEDA_artifacts/ml_m2/numeric_audit",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

