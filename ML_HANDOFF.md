# Model Handoff

## Stage

Model Stage M2: Hardware-Matched Language Model Training

Short id: ML-M2

## Status

MODEL STAGE M2 PASS.

FORMAL TRAINING COMPLETE.

## Completed

- Model Stage M1 selected the self-trained hardware-matched model as the
  primary deployment path for current RTL.
- ML-M2A froze the training/export contract.
- ML-M2B added dataset and tokenizer pipeline code.
- ML-M2C implemented the hardware-matched PyTorch causal LM.
- ML-M2D closed deterministic CPU smoke training.
- ML-M2 Formal fixed the old high-loss numeric issue before training.
- ML-M2 Formal prepared TinyStories 100000-train / 10000-validation artifacts.
- ML-M2 Formal trained the one-layer model on RTX 5080 with BF16.
- ML-M2F exported formal FP16 weights and hardware-aware traces.
- ML-M2G closed acceptance audit, summary, artifact manifest, project state,
  and handoff.

## Formal Artifacts

```text
artifact_root=D:/IC_Workspace/VEDA_artifacts/ml_m2/formal
best_checkpoint=D:/IC_Workspace/VEDA_artifacts/ml_m2/formal/checkpoints/ml_m2_formal_best.pt
best_checkpoint_sha256=cfaae278aa7fccd903b3b65041bce1b4dd91410ce3cdeacfb50e5b2b6ca933c8
last_checkpoint_sha256=968ee2d583493a816b860b05568f91c6a4ff948e0ffce82a597c151eb927fb2a
export_dir=D:/IC_Workspace/VEDA_artifacts/ml_m2/formal/exports/fp16_best
trace_dir=D:/IC_Workspace/VEDA_artifacts/ml_m2/formal/traces
```

## Key Metrics

```text
gpu=NVIDIA GeForce RTX 5080
torch=2.10.0.dev20251204+cu128
cuda_runtime=12.8
bf16_supported=true
train_stories=100000
validation_stories=10000
batch_size=1024
effective_batch_tokens=131072
epochs=3
steps=891
elapsed_seconds=121.4491955000558
initial_train_loss=7.68487024307251
best_validation_loss=3.300639416490282
perplexity=27.129980732808296
```

## Reproduction Entry Points

Expected Makefile targets exist:

```bash
make ml-m2-unit-test
make ml-m2-data-test
make ml-m2-tokenizer-test
make ml-m2-smoke-test
make ml-m2-export-test
make ml-m2-trace-test
make ml-m2-test
make ml-m2-gpu-check
make ml-m2-numeric-audit
make ml-m2-formal-data
make ml-m2-formal-train
make ml-m2-formal-eval
make ml-m2-formal-export
```

This Windows host does not have `make`, `mingw32-make`, or `nmake` installed,
so the equivalent Python runners were used directly.

## Not Completed

- Real RTL co-simulation.
- Model Stage M3.
- SmolLM2 adaptation.
- Qwen eviction experiments.
- Hardware Stage H8 integration.

## Post-Acceptance ML-Q2 Result

Model Quality Experiment Q2 completed a full-dataset fixed-architecture
benchmark without changing the ML-M2 hardware math contract.

```text
status=ML-Q2 FULL-DATASET BENCHMARK PASS
branch=ml/q2-full-dataset-benchmark
artifact_root=D:/IC_Workspace/VEDA_artifacts/ml_q2
benchmark_checkpoint=D:/IC_Workspace/VEDA_artifacts/ml_q2/benchmark/checkpoints/VEDA-HWLM-1L64-Q2.pt
benchmark_checkpoint_sha256=68b520f1322c79e568c39115809b8d623e21478af1662658cf997bf174cc9214
validation_loss=1.9365209649992428
validation_perplexity=6.934583296916738
holdout_loss=1.8095625025135131
holdout_perplexity=6.10777764135872
export_dir=D:/IC_Workspace/VEDA_artifacts/ml_q2/benchmark/exports
trace_dir=D:/IC_Workspace/VEDA_artifacts/ml_q2/benchmark/traces
```

ML-Q2 did not overwrite the accepted ML-M2 baseline checkpoint or the ML-Q1
candidate checkpoint. It did not run real RTL and did not modify functional RTL
or Hardware Stage H9 files.

## Next-Stage Cautions

- Do not modify RTL or Hardware Stage H8 files from the model branch.
- Do not commit datasets, checkpoints, tokenizer caches, FP16 weight exports,
  large traces, or credentials.
- Model Stage M3 should begin from the formal best checkpoint and compare
  PyTorch, hardware-aware bit model, and real RTL. After ML-Q2, the preferred
  fixed-architecture benchmark input is `VEDA-HWLM-1L64-Q2`.
- Generation quality is intentionally modest; use this model primarily for
  hardware-matched numeric validation and one-layer deployment experiments.
