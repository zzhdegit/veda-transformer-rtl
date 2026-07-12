# Stage 3 Summary

Status: STAGE 3 PASS.

Correctness attention accepted. Performance pipeline provisional. No PPA claim
is made.

## Scope

Stage 3 implements a first-version single-head generation attention operation:

```text
qK^T
-> scale by 1/sqrt(d)
-> online softmax reduction
-> softmax normalization
-> s'V
-> head output
```

The implementation intentionally excludes dynamic KV append, KV Cache Manager,
multi-head attention, QKV/output projections, full Transformer integration,
Voting, P&R, and formal PPA.

## Implemented RTL

- `rtl/arithmetic/fp32_exp_wrapper.sv`
- `rtl/arithmetic/fp32_recip_wrapper.sv`
- `rtl/attention/attention_score_scaler.sv`
- `rtl/attention/score_buffer.sv`
- `rtl/attention/softmax_reduction.sv`
- `rtl/attention/softmax_normalization.sv`
- `rtl/attention/single_head_attention_controller.sv`
- `rtl/attention/single_head_attention.sv`

DesignWare native FP modules are instantiated only inside project wrappers.

## Models And Tests

- `model/attention/softmax_reference.py`
- `model/attention/single_head_reference.py`
- `model/attention/single_head_cycle_model.py`
- `tb/model/test_stage3_attention.py`
- `tb/rtl/stage3/tb_fp32_exp_recip_wrappers.sv`
- `tb/rtl/stage3/tb_single_head_attention.sv`

Python bit-model coverage includes `d_head = 1, 7, 8, 9, 13, 16` and
`seq_len = 1, 2, 3, 7, 8, 15, 31, 32`.

RTL VCS coverage includes `D_HEAD=8` and `D_HEAD=16`, each with zero, uniform,
one-hot-like, and mixed input cases under output/done backpressure.

## Numeric Semantics

- Q/K/V inputs are FP16 and use the frozen Stage 1B FP16-to-FP32 policy.
- QK and SV use the frozen Stage 2 `reconfigurable_pe_core`.
- Scaling uses `fp32_mac_wrapper`: `scaled = raw_score * scale + 0`.
- EXP clamps finite inputs below `-20.0` (`32'hC1A00000`) to `+0.0`.
- Reciprocal is implemented as `1.0 / exp_sum` through `DW_fp_div` inside
  `fp32_recip_wrapper`.
- Online reduction order is serial and bit-model matched.
- SV accumulation uses Stage 2 outer-product mode and waits for real PE
  handshakes.

## Verification Results

Host:

```bash
python scripts/sim/run_stage3_tests.py
```

Result:

- 17 model tests passed.
- Python compile passed.
- Host RTL simulation skipped because host has no `vcs`.

Docker:

```bash
make stage3-rtl-sim
make stage3-lint
make stage3-synth
```

Results:

- Stage 3 RTL simulation: PASS.
- Stage 3 static lint and vlogan: PASS, diagnostics none.
- Stage 3 DC analyze/elaborate/link/check_design: PASS.
- DC checked default `D_HEAD=8`, parameterized `D_HEAD=16`, and
  `score_buffer DEPTH=4096`.

Stage 2 regression after Stage 3:

- Host `python scripts/sim/run_stage2_tests.py`: PASS.
- Docker `make stage2-rtl-sim`: PASS.
- Docker `make stage2-lint`: PASS.
- Docker `make stage2-synth`: PASS.

## RTL Cycle Counters

| D_HEAD | case | seq_len | total | qk | scale | reduction | reduction_finalize | normalization | sv | pe_stall | sfu_stall | buffer_stall | output_stall | score_peak |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | zero | 1 | 48 | 21 | 2 | 2 | 1 | 11 | 9 | 22 | 6 | 0 | 1 | 1 |
| 8 | uniform | 4 | 176 | 84 | 8 | 41 | 13 | 41 | 21 | 82 | 36 | 0 | 0 | 4 |
| 8 | onehot | 7 | 293 | 147 | 14 | 80 | 13 | 71 | 33 | 142 | 54 | 0 | 0 | 7 |
| 8 | mixed | 8 | 332 | 168 | 16 | 93 | 13 | 81 | 37 | 162 | 60 | 0 | 0 | 8 |
| 16 | zero | 1 | 77 | 41 | 2 | 2 | 1 | 11 | 19 | 44 | 6 | 0 | 1 | 1 |
| 16 | uniform | 4 | 277 | 164 | 8 | 41 | 13 | 41 | 42 | 164 | 36 | 0 | 0 | 4 |
| 16 | onehot | 7 | 466 | 287 | 14 | 80 | 13 | 71 | 66 | 284 | 54 | 0 | 0 | 7 |
| 16 | mixed | 8 | 531 | 328 | 16 | 93 | 13 | 81 | 75 | 324 | 60 | 0 | 2 | 8 |

The counters are functional RTL measurements only. They are not timing, area,
power, frequency, or PPA.

## Conclusion

STAGE 3 PASS.

The single-head attention functional loop is verified end to end. The current
pipeline is correctness-first and serial around PE/SFU dependencies; later
throughput work should be handled explicitly in a future stage.

