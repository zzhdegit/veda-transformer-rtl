# Model Handoff

## Stage

Model Stage M2: Hardware-Matched Language Model Training

Short id: ML-M2

## Status

MODEL STAGE M2 PIPELINE PASS.

FORMAL TRAINING PENDING because this environment has no CUDA GPU.

## Completed

- Model Stage M1 selected the self-trained hardware-matched model as the
  primary deployment path for current RTL.
- ML-M2A freezes the initial training/export contract.
- ML-M2B adds dataset and tokenizer pipeline code.
- ML-M2C implements the hardware-matched PyTorch causal LM.
- ML-M2D closes deterministic CPU smoke training.
- ML-M2E adds formal TinyStories workflow and records PENDING status.
- ML-M2F adds FP16 export and hardware-aware traces.
- ML-M2G adds acceptance audit, summary, and pipeline-ready tag.

## Not Completed

- Formal TinyStories training on 100000 examples.
- Model Stage M3 co-simulation against real RTL.
- Hardware Stage H8 legacy-vs-paper-array A/B integration.

## Dependencies

- Python 3.12.
- PyTorch for training and inference.
- NumPy and pytest for tests.
- Optional GPU for formal TinyStories training.

## Reproduction Entry Points

Expected make targets:

```bash
make ml-m2-unit-test
make ml-m2-data-test
make ml-m2-tokenizer-test
make ml-m2-smoke-test
make ml-m2-export-test
make ml-m2-trace-test
make ml-m2-test
```

## Next-Stage Cautions

- Do not modify RTL or Hardware Stage H8 files.
- Do not modify `model/attention/`, `model/projection/`, or
  `model/transformer/`; import them only for hardware-aware comparison.
- Do not commit datasets, checkpoints, tokenizer caches, large traces, or model
  weights.
- Do not start Model Stage M3 until ML-M2 is closed.
- Model Stage M3 may begin after user approval, but formal training remains
  pending unless GPU artifacts are produced.
