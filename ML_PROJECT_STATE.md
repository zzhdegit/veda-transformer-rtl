# Model Project State

## Current Model Stage

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
