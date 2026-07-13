# Hardware Stage H9 Verification Plan

## Model

- `tb/model/test_hw_h9_interleaved_attention.py`
- D_HEAD 8, 16, 64, 128.
- Sequence length 1, 2, 3, 7, 8, 9, 15, 16, 31, 32.
- Dense deterministic data, mixed signs, cancellation, tails, and nonuniform probabilities.
- H9 bit model vs H9 schedule model.
- H9 vs H8 metrics when add order differs.

## RTL

Focused HW-H9 RTL coverage targets:

- score buffer FIFO packet stability and overflow/underflow assertions;
- probability FIFO packet stability and overflow/underflow assertions;
- QK-SFU overlap counters;
- SFU-sV overlap counters;
- mode switch mutual exclusion;
- reset during score/probability in-flight states;
- random backpressure seeds.

## Regression

HW-H9 make targets:

- `hw-h9-test`
- `hw-h9-rtl-sim`
- `hw-h9-lint`
- `hw-h9-synth`

Stage 8, Stage 7, Stage 6, and Stage 5 regressions remain required before acceptance.
