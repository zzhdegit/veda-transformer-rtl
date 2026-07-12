# Stage 6B Summary

## Result

Stage 6B PASS.

Implemented:
- `rtl/arithmetic/fp32_to_fp16.sv`;
- `rtl/projection/projection_input_buffer.sv`;
- `rtl/projection/projection_weight_buffer.sv`;
- `rtl/projection/shared_gemv_projection_core.sv`;
- `rtl/projection/projection_controller.sv`.

## FP32-to-FP16

Policy matches Stage 6A:
- finite normal conversion;
- signed zero support;
- FP32 subnormal input FTZ;
- FP16 subnormal result FTZ;
- RNE with tie-to-even;
- mantissa/exponent carry;
- overflow saturates to signed max finite;
- NaN/Inf input is invalid and produces sign-preserving zero.

Visible interface latency is one registered cycle with initiation interval 1.
Output, metadata, and last are stable under backpressure.

## Shared GEMV

The projection GEMV path instantiates one `reconfigurable_pe_core` in
`MODE_GEMV`; it does not instantiate DesignWare directly and does not copy a
new floating-point multiply/add array.

The reduction order is:
1. FP16 operands expand through existing PE lanes.
2. Active lanes reduce through the Stage 2 balanced FP32 tree.
3. Tile sums accumulate sequentially inside the shared PE core.
4. The row result is emitted as FP32.

The first implementation computes one output row per command. The behavioral
projection input/weight buffers are correctness structures, not SRAM/PPA claims.

## Verification

Host:

```bash
python scripts/sim/run_stage6b_tests.py
```

PASS:
- Stage 5 model subset plus Stage 6 projection model tests: 16 tests.
- py_compile over model, tb, and scripts.

Docker:

```bash
make stage6b-test
make stage6b-rtl-sim
make stage6b-lint
make stage6b-synth
make stage5-rtl-sim
make stage5-lint
make stage5-synth
```

PASS:
- `stage6b-rtl-sim`: FP32-to-FP16 145 vectors; shared GEMV 16 cases.
- `stage6b-lint`: static hygiene and vlogan, no diagnostics.
- `stage6b-synth`: DC analyze/elaborate/link/check_design for
  `fp32_to_fp16`, `shared_gemv_projection_core`, and `projection_controller`.
- Stage 5 RTL/lint/DC regression remained passing.

## Counter Snapshot

From `tb_shared_gemv_projection_core`:

```text
cases=16
total_cycles=764
tile_cycles=35
pe_stall=361
output_stall=16
```

These are RTL functional counters only. No area, power, WNS, frequency, STA, or
layout conclusion is produced.

## Known Limits

- Projection memories are behavioral and reset only valid state.
- The controller is serial by output row.
- `projection_controller` is a row scheduler foundation for later QKV/W_O
  engines; Stage 6C will add Q/K/V staging and aligned Stage 5 token streaming.
