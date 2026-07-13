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

## ML-M2 Commands

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

On hosts without `make`, run the underlying Python commands:

```bash
python scripts/ml/run_ml_m2_all_tests.py
```

ML-M2 Formal artifacts are stored outside Git under
`D:/IC_Workspace/VEDA_artifacts/ml_m2/formal` on the development machine used
for the accepted formal run.
