# Model Project State

## Current Model Stage

- Stage: Model Stage M2
- Short id: ML-M2
- Name: Hardware-Matched Language Model Training
- Branch: `ml/m2-hardware-matched-model`
- Base commit: `e3b2c14a6af10cccc95f47dfadaeec2d0fc923ad`
- Last update: 2026-07-13
- Status: MODEL STAGE M2 PIPELINE PASS, FORMAL TRAINING PENDING

## Dual-Track Ownership

Hardware project state remains owned by `PROJECT_STATE.md` and `HANDOFF.md`.
Model project state is owned by `ML_PROJECT_STATE.md` and `ML_HANDOFF.md`.

Hardware Stage H8 runs independently and must not be modified by ML-M2.
Integration between the model and hardware lines is deferred to Model Stage M3.

## ML-M2 Objective

Build a reproducible training, inference, export, trace, and hardware-aware
comparison foundation for a self-trained one-layer model that matches the
accepted Stage 7 Transformer layer contract.

Formal TinyStories training is allowed to remain PENDING if no suitable GPU is
available. In that case the stage can only report:

```text
MODEL STAGE M2 PIPELINE PASS
FORMAL TRAINING PENDING
```

## ML-M2 Pipeline Result

The ML-M2 software pipeline is implemented and tested:

- data/tokenizer pipeline;
- one-layer hardware-matched PyTorch model;
- CPU smoke training;
- formal TinyStories workflow with PENDING status due no CUDA GPU;
- FP16 export;
- hardware-aware Stage 7 bit-model trace path;
- acceptance audit and summary.
