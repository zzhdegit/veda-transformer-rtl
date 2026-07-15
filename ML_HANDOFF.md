# Model Handoff

## Current Stage

Model Stage M3: Real-Weight RTL Co-Simulation and Deployment Validation

Short id: ML-M3

## Status

MODEL STAGE M3 IN PROGRESS - RTL/BIT-MODEL NUMERIC MISMATCH BLOCKED.

## Completed In ML-M3

- Confirmed model worktree `D:/IC_Workspace/VEDA_ml_m2` on branch
  `ml/m3-real-rtl-cosim`.
- Confirmed hardware read-only repo `D:/IC_Workspace/VEDA` is clean on
  `hw/h9-sfu-pe-interleaving` at tag
  `hw-h9-sfu-pe-interleaving-thesis-accepted`.
- Recomputed Q2 checkpoint SHA256:
  `68b520f1322c79e568c39115809b8d623e21478af1662658cf997bf174cc9214`.
- Recomputed tokenizer SHA256:
  `72c4100b9c923f8fc89ea563cdf18743742b87ad7cda6732606b61f50f290a1a`.
- Audited 12 Q2 export tensors and 8 RTL layer weight mappings.
- Generated real Q2 vectors for lengths 1, 2, 8, 16, and 32 under
  `D:/IC_Workspace/VEDA_artifacts/ml_m3/vectors`.
- Added model-line-only RTL testbench and VCS runner.
- Confirmed VCS/vlogan are available inside Docker container `nailong`.
- Compiled/elaborated the accepted H9 RTL for Q2 parameters:
  `N_HEAD=8`, `D_HEAD=8`, `D_MODEL=64`, `D_FFN=256`, `MAX_SEQ_LEN=128`.
- Ran one-token smoke for both H8 staged and H9 interleaved schedules.

## ML-M3 Blocker

One-token smoke failed in both schedules at the first checked real RTL final
output mismatch:

```text
CHECK_FAIL layer token=0 dim=1 got=3d4a2576 expected=3d4a2572
```

The H8 and H9 captured prefix files have the same SHA256:

```text
5adbf7b5ef5e5fbff1a767e271d852ab711ec9d829a9f7fe9125288901d4f3be
```

This means staged and interleaved schedules agree for the captured prefix, but
both differ from the current hardware-aware bit model. Per the ML-M3 gate,
length 2/8/16 RTL runs, hybrid next-token logits, and acceptance tagging were
not started.

## ML-M3 Reproduction

```bash
python scripts/ml/run_ml_m3_artifact_audit.py
python scripts/ml/run_ml_m3_vector_generation.py
python scripts/ml/run_ml_m3_vcs.py --length 1 --schedule staged --schedule interleaved --run-id smoke_len1_combined
python scripts/ml/run_ml_m3_acceptance.py
```

Key logs:

```text
D:/IC_Workspace/VEDA_artifacts/ml_m3/rtl_logs/ml_m3_staged_len_1_smoke_len1_combined.log
D:/IC_Workspace/VEDA_artifacts/ml_m3/rtl_logs/ml_m3_interleaved_len_1_smoke_len1_combined.log
```

## Not Completed In ML-M3

- length 2, 8, and 16 real RTL incremental co-simulation;
- full H8/H9 real-weight A/B cycle comparison;
- hybrid RTL-assisted next-token validation;
- accepted ML-M3 tag.

These are blocked until the RTL/bit-model numerical mismatch is resolved by a
separate hardware/reference-model fix task.

## Previous Stage

Model Stage M2: Hardware-Matched Language Model Training

Short id: ML-M2

## Previous Status

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
