# Stage Handoff

## Stage
Stage 3: Single-Head Generation Attention

## Status
STAGE 3 PASS.

Correctness attention accepted. Performance pipeline provisional. No PPA claim
is made.

## Preflight Notes
- Observed branch: `stage2-pe-core`.
- Requested branch `stage3-single-head-attention` was not present.
- Requested tag `stage2-correctness-accepted` was not present.
- Working tree was already dirty with uncommitted Stage 1B/2 deliverables.
- Stage 3 was implemented on the existing verified Stage 2 working tree without
  destructive git operations.

## Completed
- Added FP32 SFU wrappers:
  - `rtl/arithmetic/fp32_exp_wrapper.sv`
  - `rtl/arithmetic/fp32_recip_wrapper.sv`
- Added Stage 3 attention RTL:
  - `rtl/attention/attention_score_scaler.sv`
  - `rtl/attention/score_buffer.sv`
  - `rtl/attention/softmax_reduction.sv`
  - `rtl/attention/softmax_normalization.sv`
  - `rtl/attention/single_head_attention_controller.sv`
  - `rtl/attention/single_head_attention.sv`
- Added Python bit/cycle models:
  - `model/attention/softmax_reference.py`
  - `model/attention/single_head_reference.py`
  - `model/attention/single_head_cycle_model.py`
- Added Stage 3 model tests, VCS testbenches, vector generation, lint, synth
  scripts, Makefile targets, and reports under `reports/stage_03/`.

## Not Completed
- Dynamic KV append and continuous token generation.
- KV Cache Manager.
- Multi-head attention.
- QKV projection and output projection.
- Full Transformer layer.
- Voting.
- P&R, STA, formal timing closure, area, power, frequency, or PPA.

## Top Interface
`single_head_attention` is a first-version single-operation engine.

Load interface:
- `load_valid/load_ready`
- `load_kind`: `0=q`, `1=K`, `2=V`
- `load_token`
- `load_dim`
- `load_data` FP16

Command:
- `start_valid/start_ready`
- `start_seq_len`
- `start_meta`

Output stream:
- `output_valid/output_ready`
- `output_base_dim`
- `output_vector_fp32`
- `output_lane_mask`
- `output_status`
- `output_invalid`
- `output_meta`
- `output_last`

Done:
- `done_valid/done_ready`
- `done_status`
- `done_invalid`
- `done_meta`

The output is tiled by output dimension. For `D_HEAD=16, PE_NUM=8`, two output
tiles are emitted.

## Supported Range
- RTL VCS verified:
  - `PE_NUM=8, D_HEAD=8, MAX_SEQ_LEN=32`
  - `PE_NUM=8, D_HEAD=16, MAX_SEQ_LEN=32`
- Python bit model tests cover:
  - `d_head = 1, 7, 8, 9, 13, 16`
  - `seq_len = 1, 2, 3, 7, 8, 15, 31, 32`
- DC check includes:
  - default `single_head_attention`
  - `single_head_attention PE_NUM=8 D_HEAD=16 MAX_SEQ_LEN=32`
  - `score_buffer DEPTH=4096`

## Score Buffer
- Module: `rtl/attention/score_buffer.sv`
- Storage: FP32.
- Write order: token index order; assertion checks `wr_addr == valid_count`.
- Read order: controller reads token order for normalization.
- `READ_LATENCY=1` is explicit.
- Reset does not clear memory contents; it clears valid length, read count, and
  peak occupancy.
- Reads before write are assertion-checked.
- A second instance stores probabilities so the unchanged Stage 2 PE core can
  process multiple output dimension tiles for `D_HEAD > PE_NUM`.

## Scale Constants
`attention_score_scaler` uses `fp32_mac_wrapper` to compute:

```text
scaled = raw_score * scale + 0
```

Supported constants:

| D_HEAD | scale FP32 |
|---:|---|
| 1 | `32'h3F800000` |
| 7 | `32'h3EC1848F` |
| 8 | `32'h3EB504F3` |
| 9 | `32'h3EAAAAAB` |
| 13 | `32'h3E8E00D5` |
| 16 | `32'h3E800000` |
| 128 | `32'h3DB504F3` |

## Online Softmax
`softmax_reduction` is serial and correctness-first.

First score:

```text
m = score
z = 1.0
```

Later scores:

```text
m_new = max(m_old, x)
z_new = z_old * exp(m_old - m_new) + exp(x - m_new)
```

Subtraction is implemented as FP32 add with the second operand sign bit flipped,
for finite/zero inputs only.

## EXP And Reciprocal
- `fp32_exp_wrapper` wraps `DW_fp_exp`.
- Finite EXP inputs below `-20.0` (`32'hC1A00000`) clamp to `+0.0`.
- Directed EXP vectors cover:
  - `0`
  - `-0.001`
  - `-0.1`
  - `-1`
  - `-5`
  - `-10`
  - `-20`
  - below clamp input `-21`
- `fp32_recip_wrapper` computes `1.0 / x` through `DW_fp_div`.
- SFU wrappers are verified by `tb/rtl/stage3/tb_fp32_exp_recip_wrappers.sv`.

## Normalization
`softmax_normalization` computes one reciprocal:

```text
inv_sum = 1 / exp_sum
```

Then for each token:

```text
numerator_i = exp(score_i - max_final)
p_i = numerator_i * inv_sum + 0
```

Each probability carries token index and `last`; the controller writes it to the
probability buffer in token order.

## QK Scheduling
For each token:
- The controller sends `MODE_QK_INNER` tile transactions to
  `reconfigurable_pe_core`.
- `tile_first` is asserted on the first dimension tile.
- `tile_last` is asserted on the final dimension tile.
- The final tile uses the computed lane mask.
- The next tile or token waits for real PE `in_ready`.
- The raw score is consumed only when PE `out_valid && out_ready`.
- No fixed cycle guess is used for PE service time.

## SV Scheduling
For each output dimension tile:
- The probability buffer is rewound.
- For every token, the controller reads `p_i`, aligns it with the same token's
  `V_i` tile, and sends `MODE_SV_OUTER`.
- `tile_first`/`in_clear` are asserted on the first token for that output tile.
- `tile_last` is asserted on the final token for that output tile.
- The accumulated output vector is consumed only on PE
  `out_valid && out_ready`.

This supports `D_HEAD > PE_NUM` without changing `reconfigurable_pe_core`, but
it is not a throughput-optimized schedule.

## Assertions
Stage 3 RTL includes assertion/stability checks for:
- no start while busy
- score write count bounded by `seq_len`
- score write order
- no score read before written
- no buffer overflow
- no SV update before probability valid
- score/probability token index alignment
- output stable until ready
- metadata stable
- no unknown output when valid
- load payload stable under backpressure
- wrapper invalid input policy

VCS runs with assertions enabled through `-assert svaext`.

## Performance Counters
The top exposes:
- `perf_total_attention_cycles`
- `perf_qk_cycles`
- `perf_qk_pe_busy_cycles`
- `perf_scale_cycles`
- `perf_reduction_cycles`
- `perf_reduction_finalize_cycles`
- `perf_normalization_cycles`
- `perf_sv_cycles`
- `perf_pe_stall_cycles`
- `perf_sfu_stall_cycles`
- `perf_buffer_stall_cycles`
- `perf_output_stall_cycles`
- `perf_score_buffer_peak_occupancy`

Measured RTL cycles:

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

These are RTL simulation counters only.

## Reproduction
Host:

```bash
python scripts/sim/run_stage2_tests.py
python scripts/sim/run_stage3_tests.py
```

Docker:

```bash
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage2-rtl-sim'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage2-lint'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage2-synth'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage3-rtl-sim'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage3-lint'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage3-synth'
```

## Verification Results
- Host Stage 3 Python: 17 tests passed; py_compile passed.
- Docker Stage 3 RTL simulation: PASS.
- Docker Stage 3 lint: PASS.
- Docker Stage 3 DC analyze/elaborate/check_design: PASS.
- Stage 2 host and Docker regressions remain PASS after Stage 3.

## Known Limitations
- `D_HEAD` is an elaboration parameter, not a runtime input.
- Probability buffering is a correctness-first schedule for `D_HEAD > PE_NUM`;
  it does not hide normalization or SV latency.
- Softmax reduction and normalization are serial.
- Internal q/K/V memories are behavioral verification memories, not SRAM macro
  bindings.
- `score_buffer DEPTH=4096` is elaborated, but the full top was not elaborated
  with `MAX_SEQ_LEN=4096` because that would instantiate large behavioral K/V
  memories without real SRAM macros.
- No PPA, timing, area, power, WNS, frequency, STA, P&R, DRC, or LVS conclusion
  exists.

## Stage 4 Cautions
- Preserve token-major K/V layout.
- Add dynamic KV append through an explicit cache-manager interface; do not
  silently change Stage 3 q/K/V load or output formats.
- Keep probability/token index alignment explicit if adding overlap.
- Continue to respect PE `in_ready/out_valid` and drain; do not assume PE II=1.
- If Stage 4 changes K/V storage layout, PE interface, runtime `D_HEAD`, latency,
  or softmax scheduling, document it in `PROJECT_STATE.md` before implementation
  and get confirmation.

## Recommended Commit Message
```text
stage3: implement verified single-head generation attention
```
