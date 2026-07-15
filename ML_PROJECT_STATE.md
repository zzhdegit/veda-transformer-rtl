# Model Project State

## Current Model Stage

- Stage: Model Stage M3
- Short id: ML-M3
- Name: Real-Weight RTL Co-Simulation and Deployment Validation
- Branch: `ml/m3-real-rtl-cosim`
- Base tag: `ml-q2-full-dataset-benchmark-accepted`
- Last update: 2026-07-15
- Status: MODEL STAGE M3 IN PROGRESS - RTL/BIT-MODEL NUMERIC MISMATCH BLOCKED

ML-M3 consumed the frozen Q2 benchmark checkpoint and generated real Q2
vectors for lengths 1, 2, 8, 16, and 32. Q2 artifact audit, tokenizer SHA,
export tensor audit, and weight mapping audit passed. H8 staged and H9
interleaved RTL both compiled/elaborated for the Q2 H8/D8 layer configuration,
but one-token smoke failed before multi-token RTL co-simulation:

```text
CHECK_FAIL layer token=0 dim=1 got=3d4a2576 expected=3d4a2572
```

Both H8 and H9 produced the same captured prefix SHA
`5adbf7b5ef5e5fbff1a767e271d852ab711ec9d829a9f7fe9125288901d4f3be`,
so the current blocker is common RTL-vs-bit-model numerical mismatch, not a
staged/interleaved divergence. ML-M3 did not modify hardware files, did not run
multi-token RTL after the one-token gate failed, and did not invoke PDK, STA,
P&R, or PPA.

## Previous Model Stage

- Stage: Model Stage M2
- Short id: ML-M2
- Name: Hardware-Matched Language Model Training
- Branch: `ml/m2-hardware-matched-model`
- Base commit: `e3b2c14a6af10cccc95f47dfadaeec2d0fc923ad`
- Last update: 2026-07-13
- Status: MODEL STAGE M2 PASS, FORMAL TRAINING COMPLETE

## Dual-Track Ownership

Hardware project state remains owned by `PROJECT_STATE.md` and `HANDOFF.md`.
Model project state is owned by `ML_PROJECT_STATE.md` and `ML_HANDOFF.md`.

Hardware Stage H8 runs independently and was not modified by ML-M2. Integration
between the model and hardware lines is deferred to Model Stage M3.

## ML-M2 Result

The ML-M2 software pipeline and formal training closure are complete:

- hardware-matched one-layer PyTorch model;
- deterministic TinyStories data/tokenizer pipeline;
- pretraining numeric audit and regression tests;
- RTX 5080 CUDA/BF16 environment audit;
- formal TinyStories training on 100000 train stories and 10000 validation
  stories;
- best and last checkpoints in artifact storage;
- FP16 weight export for 12 tensors;
- hardware-aware Stage 7 bit-model traces for prompt lengths 1, 2, 8, and 16;
- artifact manifests, SHA256 records, acceptance audit, summary, and handoff.

## Formal Checkpoint

```text
best_checkpoint=D:/IC_Workspace/VEDA_artifacts/ml_m2/formal/checkpoints/ml_m2_formal_best.pt
best_checkpoint_sha256=cfaae278aa7fccd903b3b65041bce1b4dd91410ce3cdeacfb50e5b2b6ca933c8
best_validation_loss=3.300639416490282
perplexity=27.129980732808296
```

## Next Stage

Model Stage M3 may begin after user approval. It should consume the ML-M2
formal checkpoint/export/trace artifacts for PyTorch / bit model / RTL
co-simulation. ML-M2 did not start real RTL co-simulation.

## Post-Acceptance Quality Experiments

### Model Quality Experiment Q2

- Short id: ML-Q2
- Name: Full-Dataset Hardware Benchmark Training
- Branch: `ml/q2-full-dataset-benchmark`
- Last update: 2026-07-13
- Status: ML-Q2 FULL-DATASET BENCHMARK PASS

ML-Q2 kept the accepted ML-M2 hardware-matched architecture and BPE-2048
tokenizer unchanged, used the full official TinyStories train split, and
selected `VEDA-HWLM-1L64-Q2` as the internal fixed-architecture benchmark
checkpoint for a later Model Stage M3.

```text
benchmark_checkpoint=D:/IC_Workspace/VEDA_artifacts/ml_q2/benchmark/checkpoints/VEDA-HWLM-1L64-Q2.pt
benchmark_checkpoint_sha256=68b520f1322c79e568c39115809b8d623e21478af1662658cf997bf174cc9214
validation_loss=1.9365209649992428
holdout_loss=1.8095625025135131
validation_perplexity=6.934583296916738
holdout_perplexity=6.10777764135872
```

ML-Q2 did not run real RTL co-simulation, did not start Model Stage M3, and
did not modify functional RTL or Hardware Stage H9 files.
