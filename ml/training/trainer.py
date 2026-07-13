"""Minimal trainer used by ML-M2 smoke and formal workflows."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, TensorDataset

from ml.architecture.causal_lm import HardwareMatchedCausalLM
from ml.architecture.config import HardwareMatchedConfig
from ml.data.sequence_builder import SequenceBatch
from ml.training.optimizer import build_optimizer
from ml.training.reproducibility import set_seed
from ml.training.scheduler import build_scheduler


@dataclass(frozen=True)
class TrainingMetrics:
    initial_train_loss: float
    final_train_loss: float
    validation_loss: float
    perplexity: float
    steps: int
    epochs: float
    train_examples: int
    validation_examples: int
    elapsed_seconds: float
    device: str
    no_nan_inf: bool
    grad_norm: float = 0.0


def make_dataset(batch: SequenceBatch) -> TensorDataset:
    x = torch.tensor(batch.input_ids, dtype=torch.long)
    y = torch.tensor(batch.labels, dtype=torch.long)
    return TensorDataset(x, y)


@torch.no_grad()
def evaluate_loss(model: HardwareMatchedCausalLM, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        out = model(input_ids, labels=labels)
        losses.append(float(out["loss"].detach().cpu()))
    return sum(losses) / len(losses) if losses else float("nan")


def train_for_steps(
    config: HardwareMatchedConfig,
    train_batch: SequenceBatch,
    validation_batch: SequenceBatch,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float,
    seed: int,
    device: str = "cpu",
) -> tuple[HardwareMatchedCausalLM, TrainingMetrics]:
    set_seed(seed)
    torch_device = torch.device(device)
    model = HardwareMatchedCausalLM(config).to(torch_device)
    train_ds = make_dataset(train_batch)
    val_ds = make_dataset(validation_batch)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    optimizer = build_optimizer(model, learning_rate=learning_rate, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer, total_steps=steps)
    iterator = iter(train_loader)
    initial_loss = None
    final_loss = None
    no_nan_inf = True
    last_grad_norm = 0.0
    start = time.perf_counter()
    model.train()
    for step in range(steps):
        try:
            input_ids, labels = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            input_ids, labels = next(iterator)
        input_ids = input_ids.to(torch_device)
        labels = labels.to(torch_device)
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids, labels=labels)
        loss = out["loss"]
        if not torch.isfinite(loss):
            no_nan_inf = False
            break
        if initial_loss is None:
            initial_loss = float(loss.detach().cpu())
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        last_grad_norm = float(grad_norm.detach().cpu()) if hasattr(grad_norm, "detach") else float(grad_norm)
        optimizer.step()
        scheduler.step()
        final_loss = float(loss.detach().cpu())
    elapsed = time.perf_counter() - start
    validation_loss = evaluate_loss(model, val_loader, torch_device)
    perplexity = math.exp(validation_loss) if math.isfinite(validation_loss) and validation_loss < 50 else float("inf")
    epochs = (steps * batch_size) / max(len(train_ds), 1)
    metrics = TrainingMetrics(
        initial_train_loss=float(initial_loss if initial_loss is not None else "nan"),
        final_train_loss=float(final_loss if final_loss is not None else "nan"),
        validation_loss=validation_loss,
        perplexity=perplexity,
        steps=steps,
        epochs=epochs,
        train_examples=len(train_ds),
        validation_examples=len(val_ds),
        elapsed_seconds=elapsed,
        device=str(torch_device),
        no_nan_inf=no_nan_inf,
        grad_norm=last_grad_norm,
    )
    return model.cpu(), metrics
