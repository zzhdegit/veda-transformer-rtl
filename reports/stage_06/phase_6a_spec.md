# Stage 6A Specification Freeze

## Result

Stage 6A freezes the Projection-Integrated Multi-Head Attention scope, numeric
boundaries, matrix layout, index mapping, and Python bit-model framework.

## Frozen Decisions

- Stage 6 scope is projection-integrated MHA only.
- Norm, residual, FFN, activation, full Transformer layer, P&R, STA, SRAM macro
  binding, and PPA are deferred to Stage 7 or Stage 8.
- `D_MODEL = N_HEAD * D_HEAD`.
- Weight layout is `weight[matrix_kind][output_index][input_index]`.
- Q/K/V projection order is Q rows, then K rows, then V rows.
- `projection_output_index = head * D_HEAD + dim`.
- `concat_index = head * D_HEAD + dim`.
- GEMV reduction order reuses Stage 2: balanced tree per tile, sequential tile
  accumulation.
- Q/K/V projection results are FP32, then quantized to FP16 before Stage 5.
- Stage 5 outputs are FP32, written to a concat buffer, then quantized to FP16
  before `W_O`.
- Final Stage 6 output remains FP32.

## FP32-to-FP16 Policy

- RNE rounding.
- signed zero is sign-preserving.
- FP32 subnormal input flushes to signed zero.
- FP16 subnormal result flushes to signed zero.
- overflow saturates to signed maximum finite FP16.
- NaN/Inf inputs are illegal, return sign-preserving zero, and set invalid.

## Model Files

- `model/projection/fp32_fp16_reference.py`
- `model/projection/gemv_reference.py`
- `model/projection/projection_reference.py`
- `model/projection/projection_mha_reference.py`

## Verification

Stage 6A model tests cover:

- D_MODEL 8, 16, and 32;
- head/dim/output-index mapping;
- weight row-major address mapping;
- FP16 input and weight layouts;
- Q/K/V split and stream order;
- concat order;
- RNE tie-to-even vectors;
- overflow, underflow/FTZ, NaN, and Inf policy.

Run:

```bash
make stage6a-test
```

## Executed Checks

Host:

```bash
python scripts/sim/run_stage6a_tests.py
```

PASS:
- Stage 5 model regression subset plus Stage 6A projection tests: 16 tests.
- py_compile over model, tb, and scripts.

Docker:

```bash
make stage6a-test
make stage5-rtl-sim
make stage5-lint
make stage5-synth
```

PASS:
- `make stage6a-test`: fallback runner, 16 tests, py_compile with Python 3.6
  compatibility skips for existing Python 3.7+ files.
- `make stage5-rtl-sim`: cache manager plus H1/D8, H2/D8, H4/D8, H2/D16.
- `make stage5-lint`: static hygiene and vlogan, no diagnostics.
- `make stage5-synth`: DC analyze/elaborate/link/check_design, no PPA.
