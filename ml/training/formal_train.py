"""Formal RTX 5080 TinyStories training for ML-M2."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader, TensorDataset

from ml.architecture.causal_lm import HardwareMatchedCausalLM
from ml.architecture.config import HardwareMatchedConfig
from ml.data.dataset_hash import sha256_file
from ml.data.dataset_manifest import artifact_root
from ml.inference.generate import generate_text
from ml.tokenizer.load_tokenizer import SimpleBPETokenizer
from ml.training.optimizer import build_optimizer
from ml.training.reproducibility import set_seed
from ml.training.scheduler import build_scheduler


@dataclass(frozen=True)
class FormalTrainingConfig:
    batch_size: int = 512
    epochs: int = 3
    max_epochs: int = 5
    learning_rate: float = 3.0e-4
    minimum_lr: float = 3.0e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1.0e-8
    grad_clip_norm: float = 1.0
    warmup_fraction: float = 0.02
    validation_interval: int = 100
    checkpoint_interval: int = 250
    dtype: str = "bf16_if_supported"
    use_fused_adamw: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    seed: int = 20260713


def formal_root() -> Path:
    return artifact_root() / "formal"


def _load_manifest(root: Path) -> dict:
    return json.loads((root / "data" / "formal_data_manifest.json").read_text(encoding="utf-8"))


def _load_tensor_dataset(path: str | Path) -> TensorDataset:
    payload = torch.load(Path(path), map_location="cpu")
    return TensorDataset(payload["input_ids"].long(), payload["labels"].long())


def _loader(dataset: TensorDataset, batch_size: int, shuffle: bool, seed: int, cfg: FormalTrainingConfig) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "pin_memory": cfg.pin_memory and torch.cuda.is_available(),
        "num_workers": cfg.num_workers,
        "generator": torch.Generator().manual_seed(seed),
    }
    if cfg.num_workers > 0:
        kwargs["persistent_workers"] = cfg.persistent_workers
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


@torch.no_grad()
def evaluate_loss(model: HardwareMatchedCausalLM, loader: DataLoader, device: torch.device, use_bf16: bool) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for input_ids, labels in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16 and device.type == "cuda"):
            loss = model(input_ids, labels=labels)["loss"]
        total_loss += float(loss.detach().cpu())
        total_batches += 1
    return total_loss / max(total_batches, 1)


def _gpu_utilization() -> int | None:
    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        return int(result.splitlines()[0].strip())
    except Exception:
        return None


def _cpu_percent() -> float | None:
    try:
        import psutil  # type: ignore

        return float(psutil.cpu_percent(interval=0.1))
    except Exception:
        return None


def _step(
    model: HardwareMatchedCausalLM,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_bf16: bool,
    grad_clip_norm: float,
) -> tuple[float, float]:
    input_ids, labels = batch
    input_ids = input_ids.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16 and device.type == "cuda"):
        loss = model(input_ids, labels=labels)["loss"]
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite loss during formal training")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
    return float(loss.detach().cpu()), float(grad_norm.detach().cpu() if hasattr(grad_norm, "detach") else grad_norm)


def _cycle(loader: Iterable):
    while True:
        for batch in loader:
            yield batch


def benchmark_batches(
    root: str | Path | None = None,
    candidates: list[int] | None = None,
    warmup_steps: int = 2,
    measure_steps: int = 5,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ML-M2 formal benchmark")
    root_path = Path(root) if root else formal_root()
    manifest = _load_manifest(root_path)
    dataset = _load_tensor_dataset(manifest["packing"]["train_tensor"]["path"])
    cfg = FormalTrainingConfig()
    model_cfg = HardwareMatchedConfig(vocab_size=manifest["tokenizer"]["vocab_size"], context_length=manifest["packing"]["context_length"])
    use_bf16 = torch.cuda.is_bf16_supported()
    results = []
    for batch_size in candidates or [64, 128, 256, 512, 1024]:
        try:
            set_seed(model_cfg.seed)
            model = HardwareMatchedCausalLM(model_cfg).cuda()
            optimizer = build_optimizer(model, learning_rate=3.0e-4, weight_decay=0.1, fused=cfg.use_fused_adamw)
            loader = _loader(dataset, batch_size=batch_size, shuffle=True, seed=model_cfg.seed, cfg=cfg)
            iterator = _cycle(loader)
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for _ in range(warmup_steps):
                _step(model, next(iterator), optimizer, torch.device("cuda"), use_bf16, cfg.grad_clip_norm)
            torch.cuda.synchronize()
            measure_start = time.perf_counter()
            last_loss = 0.0
            for _ in range(measure_steps):
                last_loss, _ = _step(model, next(iterator), optimizer, torch.device("cuda"), use_bf16, cfg.grad_clip_norm)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - measure_start
            tokens = batch_size * model_cfg.context_length * measure_steps
            results.append(
                {
                    "batch_size": batch_size,
                    "status": "ok",
                    "tokens_per_second": tokens / max(elapsed, 1e-9),
                    "sequences_per_second": (batch_size * measure_steps) / max(elapsed, 1e-9),
                    "step_time_seconds": elapsed / measure_steps,
                    "last_loss": last_loss,
                    "warmup_elapsed_seconds": measure_start - start,
                    "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()),
                    "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved()),
                    "gpu_utilization_percent": _gpu_utilization(),
                    "cpu_utilization_percent": _cpu_percent(),
                    "data_loader_wait_seconds": None,
                    "oom": False,
                }
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            results.append({"batch_size": batch_size, "status": "oom", "error": str(exc), "oom": True})
    ok = [item for item in results if item.get("status") == "ok"]
    selected = max(ok, key=lambda item: item["tokens_per_second"]) if ok else None
    manifest_out = {
        "stage": "ML-M2 Formal",
        "dtype": "bf16" if use_bf16 else "fp32",
        "torch_compile": "not_enabled_windows_eager_selected",
        "compile_reason": "eager path retained unless a separate benchmark proves torch.compile improves tokens/s with identical outputs",
        "results": results,
        "selected_batch_size": selected["batch_size"] if selected else None,
    }
    out_path = root_path / "training" / "gpu_throughput_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest_out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_out


def run_formal_training(root: str | Path | None = None, cfg: FormalTrainingConfig | None = None) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ML-M2 formal training")
    cfg = cfg or FormalTrainingConfig()
    root_path = Path(root) if root else formal_root()
    manifest = _load_manifest(root_path)
    train_ds = _load_tensor_dataset(manifest["packing"]["train_tensor"]["path"])
    val_ds = _load_tensor_dataset(manifest["packing"]["validation_tensor"]["path"])
    model_cfg = HardwareMatchedConfig(
        vocab_size=manifest["tokenizer"]["vocab_size"],
        context_length=manifest["packing"]["context_length"],
        seed=cfg.seed,
    )
    set_seed(model_cfg.seed)
    device = torch.device("cuda")
    use_bf16 = cfg.dtype == "bf16" or (cfg.dtype == "bf16_if_supported" and torch.cuda.is_bf16_supported())
    train_loader = _loader(train_ds, batch_size=cfg.batch_size, shuffle=True, seed=cfg.seed, cfg=cfg)
    val_loader = _loader(val_ds, batch_size=cfg.batch_size, shuffle=False, seed=cfg.seed, cfg=cfg)
    steps_per_epoch = math.ceil(len(train_ds) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = max(1, int(total_steps * cfg.warmup_fraction))
    min_lr_ratio = cfg.minimum_lr / cfg.learning_rate
    model = HardwareMatchedCausalLM(model_cfg).to(device)
    optimizer = build_optimizer(
        model,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        fused=cfg.use_fused_adamw,
    )
    scheduler = build_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
        schedule="cosine",
    )
    training_dir = root_path / "training"
    ckpt_dir = root_path / "checkpoints"
    training_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    untrained_validation_loss = evaluate_loss(model, val_loader, device, use_bf16)
    iterator = _cycle(train_loader)
    history = []
    best_validation = float("inf")
    best_path = ckpt_dir / "ml_m2_formal_best.pt"
    last_path = ckpt_dir / "ml_m2_formal_last.pt"
    no_nan_inf = True
    initial_train_loss = None
    final_train_loss = None
    last_grad_norm = 0.0
    start = time.perf_counter()
    for step in range(1, total_steps + 1):
        try:
            loss_value, last_grad_norm = _step(model, next(iterator), optimizer, device, use_bf16, cfg.grad_clip_norm)
        except FloatingPointError:
            no_nan_inf = False
            raise
        scheduler.step()
        if initial_train_loss is None:
            initial_train_loss = loss_value
        final_train_loss = loss_value
        should_eval = step == 1 or step == total_steps or step % cfg.validation_interval == 0
        if should_eval:
            validation_loss = evaluate_loss(model, val_loader, device, use_bf16)
            record = {
                "step": step,
                "epoch": step / max(steps_per_epoch, 1),
                "train_loss": loss_value,
                "validation_loss": validation_loss,
                "perplexity": math.exp(validation_loss) if validation_loss < 50 else float("inf"),
                "lr": float(scheduler.get_last_lr()[0]),
                "grad_norm": last_grad_norm,
            }
            history.append(record)
            if validation_loss < best_validation:
                best_validation = validation_loss
                _save_formal_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    step,
                    model_cfg,
                    cfg,
                    record,
                    manifest,
                )
        if step == total_steps or step % cfg.checkpoint_interval == 0:
            _save_formal_checkpoint(
                last_path,
                model,
                optimizer,
                scheduler,
                step,
                model_cfg,
                cfg,
                history[-1] if history else {"step": step, "train_loss": loss_value},
                manifest,
            )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    final_validation_loss = evaluate_loss(model, val_loader, device, use_bf16)
    if final_validation_loss < best_validation:
        best_validation = final_validation_loss
        _save_formal_checkpoint(
            best_path,
            model,
            optimizer,
            scheduler,
            total_steps,
            model_cfg,
            cfg,
            {"step": total_steps, "validation_loss": final_validation_loss, "train_loss": final_train_loss},
            manifest,
        )
    _save_formal_checkpoint(
        last_path,
        model,
        optimizer,
        scheduler,
        total_steps,
        model_cfg,
        cfg,
        {"step": total_steps, "validation_loss": final_validation_loss, "train_loss": final_train_loss},
        manifest,
    )
    tokenizer = SimpleBPETokenizer.load(manifest["tokenizer"]["tokenizer_json"])
    model_cpu = model.cpu().eval()
    prompts = json.loads(Path(manifest["test_prompts"]["path"]).read_text(encoding="utf-8"))["prompts"]
    samples = [
        {"prompt": prompt, "greedy": generate_text(model_cpu, tokenizer, prompt, max_new_tokens=24)}
        for prompt in prompts[:5]
    ]
    samples_path = training_dir / "generation_samples.json"
    samples_path.write_text(json.dumps(samples, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tokens_per_second = (len(train_ds) * model_cfg.context_length * cfg.epochs) / max(elapsed, 1e-9)
    metrics = {
        "stage": "ML-M2 Formal",
        "status": "FORMAL_TRAINING_COMPLETE",
        "device": torch.cuda.get_device_name(0),
        "dtype": "bf16" if use_bf16 else "fp32",
        "train_examples": manifest["subsets"]["train_stories"],
        "validation_examples": manifest["subsets"]["validation_stories"],
        "train_packed_sequences": len(train_ds),
        "validation_packed_sequences": len(val_ds),
        "total_training_tokens": int(len(train_ds) * model_cfg.context_length * cfg.epochs),
        "batch_size": cfg.batch_size,
        "effective_batch_tokens": int(cfg.batch_size * model_cfg.context_length),
        "epochs": cfg.epochs,
        "steps": total_steps,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens_per_second,
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved()),
        "gpu_utilization_percent": _gpu_utilization(),
        "cpu_utilization_percent": _cpu_percent(),
        "untrained_validation_loss": untrained_validation_loss,
        "initial_train_loss": float(initial_train_loss if initial_train_loss is not None else "nan"),
        "final_train_loss": float(final_train_loss if final_train_loss is not None else "nan"),
        "best_validation_loss": best_validation,
        "final_validation_loss": final_validation_loss,
        "perplexity": math.exp(best_validation) if best_validation < 50 else float("inf"),
        "train_validation_gap": float(final_train_loss - best_validation) if final_train_loss is not None else float("nan"),
        "no_nan_inf": no_nan_inf,
        "last_grad_norm": last_grad_norm,
        "history": history,
        "generation_samples": str(samples_path),
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint": str(last_path),
        "last_checkpoint_sha256": sha256_file(last_path),
        "data_manifest": str(root_path / "data" / "formal_data_manifest.json"),
    }
    metrics_path = training_dir / "formal_training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def _save_formal_checkpoint(
    path: Path,
    model: HardwareMatchedCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    model_cfg: HardwareMatchedConfig,
    training_cfg: FormalTrainingConfig,
    metrics: dict,
    data_manifest: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": None,
        "config": model_cfg.to_json_dict(),
        "training_config": asdict(training_cfg),
        "metrics": metrics,
        "data_manifest": data_manifest,
    }
    torch.save(payload, path)
    manifest = {"path": str(path), "sha256": sha256_file(path), "step": step, "metrics": metrics}
    path.with_suffix(path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(formal_root()))
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--validation-interval", type=int, default=100)
    args = parser.parse_args()
    if args.benchmark_only:
        print(json.dumps(benchmark_batches(args.root), indent=2, sort_keys=True))
        return
    cfg = FormalTrainingConfig(batch_size=args.batch_size, epochs=args.epochs, validation_interval=args.validation_interval)
    print(json.dumps(run_formal_training(args.root, cfg), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
