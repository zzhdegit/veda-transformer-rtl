# Stage 6D Projected Multi-Head Attention Summary

## Scope

Stage 6D integrates the verified Stage 6C `qkv_projection_engine` with the accepted Stage 5 shared multi-head generation engine in `projected_multi_head_attention`.

This phase accepts FP16 hidden-state tokens, applies shared GEMV Q/K/V projection with FP32-to-FP16 QKV quantization, then streams aligned per-head Q/K/V into Stage 5. The Stage 5 numerical path, K/V cache commit rule, all-head atomicity, current-token causal attention, output head ordering, and cache-full behavior are preserved.

Stage 6D intentionally does not implement head concat or W_O output projection. Those are Stage 6E/6F scope.

## RTL Added

- `rtl/attention/projected_multi_head_attention.sv`

The top level instantiates:

- `rtl/projection/qkv_projection_engine.sv`
- `rtl/attention/multi_head_generation_engine.sv`

The wrapper gates new hidden tokens while a projected-attention transaction is active, holds QKV streams stable under Stage 5 backpressure, and exposes cumulative performance counters for Q, K, V, QKV quantization, attention, stalls, generated steps, and peak valid sequence length.

## Verification Commands

- `python scripts/sim/run_stage6d_tests.py`
- `make stage6d-test`
- `make stage6d-rtl-sim`
- `make stage6d-lint`
- `make stage6d-synth`
- `make stage5-rtl-sim`
- `make stage5-lint`
- `make stage5-synth`

## Results

- Python model and vector generation: PASS
- VCS projected MHA RTL simulations: PASS for H1/D8, H2/D8, H4/D8, and H2/D16
- Static lint/vlogan hygiene: PASS
- DC analyze/elaborate/link/check_design: PASS for H1/D8, H2/D8, H4/D8, and H2/D16
- Stage 5 regression after integration: PASS

## Cycle Counter Samples

The VCS report records per-token cumulative counters. Final cache-full steps confirm no valid sequence length increment beyond MAX_SEQ_LEN.

| Config | Generation steps | Final peak seq | Final total cycles | Final Q cycles | Final K cycles | Final V cycles | Final QKV quant cycles | Final attention cycles |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H1 D8 | 8 | 8 | 7953 | 1674 | 1674 | 1674 | 432 | 2777 |
| H2 D8 | 8 | 8 | 24482 | 6210 | 6210 | 6210 | 864 | 5555 |
| H4 D8 | 8 | 8 | 83465 | 23922 | 23922 | 23922 | 1728 | 11113 |
| H2 D16 | 8 | 8 | 82138 | 23922 | 23922 | 23922 | 1728 | 9803 |

## Test Notes

The Stage 6D RTL vectors use sparse exact FP16 projection matrices that map deterministic hidden-state dimensions into Q/K and preserve V through identity-style rows. This keeps the integration test inside the already accepted Stage 5 numerical envelope while still checking QKV projection ordering, FP32-to-FP16 quantization, head/dim mapping, Stage 5 transaction flow, repeated generation, backpressure, and cache-full behavior.

Richer projection matrices are covered at the Stage 6C QKV boundary and will be reintroduced in Stage 6E/6F around the final projection-integrated top after the output projection bit model is closed.

## Limitations

- Behavioral weight and cache memories remain structural verification models, not SRAM macros.
- No formal area, power, frequency, WNS, STA, or P&R claims are made.
- Head concat and W_O projection are not present in this phase.
