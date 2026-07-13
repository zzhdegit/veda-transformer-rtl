# ML Workstream

This directory owns the model-side workstream for VEDA.

Hardware project status remains in `PROJECT_STATE.md` and `HANDOFF.md`.
Model project status is tracked in `ML_PROJECT_STATE.md` and `ML_HANDOFF.md`.

## Stage Naming

Model stages must use the `Model Stage M*` naming scheme. The current stage is:

```text
Model Stage M2: Hardware-Matched Language Model Training
Short id: ML-M2
Branch: ml/m2-hardware-matched-model
```

Hardware work in parallel is called `Hardware Stage H8`. ML-M2 does not modify
RTL, hardware Stage H8 files, or the accepted bit-accurate reference model under
`model/`.

## Directory Policy

The repository `model/` directory remains the RTL bit-accurate reference model.
PyTorch training, tokenization, inference, export, tracing, co-simulation, and
evaluation code lives under `ml/`.

Large data and model artifacts must stay outside Git. Use:

```text
VEDA_ML_DATA_ROOT
VEDA_ML_ARTIFACT_ROOT
VEDA_HF_CACHE
```

Recommended local layout:

```text
D:/IC_Workspace/VEDA_artifacts/
|-- datasets/
|-- tokenizers/
|-- checkpoints/
|-- traces/
|-- exports/
`-- rtl_vectors/
```

