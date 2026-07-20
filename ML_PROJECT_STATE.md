# Model Project State

## Current Model Stage

- Stage: Model Stage M3
- Short id: ML-M3
- Name: Real-Weight RTL Co-Simulation and Deployment Validation
- Branch: `ml/m3-real-rtl-cosim`
- Base tag: `ml-q2-full-dataset-benchmark-accepted`
- Last update: 2026-07-20
- Status: MODEL STAGE M3 PASS

ML-M3 consumed the frozen Q2 benchmark checkpoint and generated real Q2 vectors
for lengths 1, 2, 8, 16, and 32. Q2 artifact audit, tokenizer SHA, export
tensor audit, and weight mapping audit passed. The accepted hardware baseline
is the repaired H9 tag:

```text
tag=hw-h9-real-weight-numeric-repair-accepted
commit=a54e608a8dc7e63c7e5dd342f8b893bb1e0b7485
```

The repaired H8 staged and H9 interleaved RTL both compile and pass bit-exact
real-weight co-simulation against the hardware-aware bit model for lengths 1,
2, 8, and 16 in both no-stall and deterministic output+done stall modes. H8
and H9 captured outputs are identical for every accepted core case. Length32
no-stall also passes as an extended co-simulation case.

ML-M3 also closed:

- software full-vs-incremental reference comparison;
- valid_seq_len progression 1..N for real RTL runs;
- output lane/tile, done, metadata, and duplicate/missing output checks;
- 9 categories of real RTL internal node comparison per schedule;
- H8/H9 real-weight cycle comparison;
- 3 hybrid next-token cases and one continuous 2-step prediction;
- forbidden-path and hardware read-only audits.

No hardware source files, checkpoints, tokenizer files, PDK, STA, P&R, PPA, or
Hardware Stage H10 flow were modified or invoked.

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

## ML-M3 Acceptance Result

Model Stage M3 is accepted on branch `ml/m3-real-rtl-cosim`. Acceptance evidence
is recorded in `reports/ml_m3/acceptance_audit.md`,
`reports/ml_m3/summary.md`, and
`D:/IC_Workspace/VEDA_artifacts/ml_m3/manifests/acceptance.json`.

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
