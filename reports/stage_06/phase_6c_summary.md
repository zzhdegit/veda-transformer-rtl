# Stage 6C QKV Projection Summary

## Scope

Stage 6C implements the dimension-serial hidden-state input path, shared W_Q/W_K/W_V loading, serial Q then K then V projection scheduling, FP32-to-FP16 QKV quantization, staging, and head/dim ordered QKV output stream.

The implementation reuses the Stage 6B `projection_controller`, `shared_gemv_projection_core`, and `fp32_to_fp16` quantizer. It does not instantiate a second floating-point PE array and does not alter Stage 5 attention RTL.

## Implemented RTL

- `rtl/projection/qkv_staging_buffer.sv`
- `rtl/projection/qkv_projection_engine.sv`
- Updated `rtl/projection/projection_input_buffer.sv` so same-cycle final-dimension `load_commit` observes the accepted last input beat.

## Interface and Ordering

- Hidden input order is strict dimension order `0..D_MODEL-1`.
- Weight layout remains output-row-major: `weight[kind][output_index][input_index]`.
- Q/K/V projection order is fixed: all Q rows, all K rows, all V rows.
- QKV stream order is `head * D_HEAD + dim`.
- Output fields are aligned per stream element: `qkv_head`, `qkv_dim`, `qkv_q_fp16`, `qkv_k_fp16`, `qkv_v_fp16`, `qkv_last_dim`, `qkv_last_head`, `qkv_meta`.

## Verification Commands

Host:

```text
python scripts/sim/run_stage6c_tests.py
```

Docker:

```text
make stage6c-test
make stage6c-rtl-sim
make stage6c-lint
make stage6c-synth
make stage5-rtl-sim
make stage5-lint
make stage5-synth
```

## Results

- Python model tests: PASS, 16 tests.
- Stage 6C RTL simulation: PASS for H1/D8, H2/D8, H4/D8, H2/D16.
- Stage 6C lint/vlogan: PASS, no diagnostics.
- Stage 6C DC analyze/elaborate/link/check_design: PASS for H1/D8, H2/D8, H4/D8, H2/D16.
- Stage 5 regression: PASS for RTL sim, lint, and DC elaboration.

## RTL Cycle Markers

```text
STAGE6C_QKV_PROJECTION_PASS N_HEAD=1 D_HEAD=8 q=372 k=372 v=372 quant=96 pe_stall=0 output_stall=6
STAGE6C_QKV_PROJECTION_PASS N_HEAD=2 D_HEAD=8 q=1380 k=1380 v=1380 quant=192 pe_stall=1824 output_stall=16
STAGE6C_QKV_PROJECTION_PASS N_HEAD=4 D_HEAD=8 q=5316 k=5316 v=5316 quant=384 pe_stall=10944 output_stall=28
STAGE6C_QKV_PROJECTION_PASS N_HEAD=2 D_HEAD=16 q=5316 k=5316 v=5316 quant=384 pe_stall=10944 output_stall=28
```

## Physical Status

Projection weights and staging buffers remain behavioral arrays for correctness closure. The DC step is structural elaboration only; it is not a PPA result and does not imply SRAM macro binding, timing closure, area, power, WNS, frequency, or P&R status.
