# Model Handoff

## Stage

Model Stage M2: Hardware-Matched Language Model Training

Short id: ML-M2

## Status

ML-M2 is beginning from Model Stage M1 commit
`e3b2c14a6af10cccc95f47dfadaeec2d0fc923ad`.

## Completed

- Model Stage M1 selected the self-trained hardware-matched model as the
  primary deployment path for current RTL.
- ML-M2A freezes the initial training/export contract.

## Not Completed

- Dataset/tokenizer pipeline.
- PyTorch architecture and unit tests.
- CPU smoke training.
- Formal TinyStories training.
- FP16 export and trace generation.
- Hardware-aware comparison.
- ML-M2 acceptance audit.

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

