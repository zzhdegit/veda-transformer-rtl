# Project State

## Current Stage

- Stage: 6
- Status: STAGE 6 PASS
- Branch: `stage6-projection-mha`
- Last update: 2026-07-13

projection-integrated multi-head attention correctness accepted.

throughput, physical memory, and timing pipeline provisional.

Stage 6 implements projection-integrated multi-head attention only. It does not
implement RMSNorm, LayerNorm, residual paths, FFN, activation functions, full
Transformer layer integration, SRAM macro binding, STA, layout, or PPA.

## Accepted Stage 6 Scope

```text
hidden FP16
-> Q/K/V projection
-> FP32-to-FP16 Q/K/V quantization
-> Stage 5 multi-head current-token causal attention
-> streamed FP32 head concat
-> FP32-to-FP16 concat quantization
-> FP16 concat buffer
-> W_O output projection
-> final tiled FP32 MHA output
```

Frozen relations and layout:

- `D_MODEL = N_HEAD * D_HEAD`
- Q/K/V/W_O weights use `weight[kind][output_index][input_index]`
- concat index is `output_head * D_HEAD + output_base_dim + lane_index`
- W_O computes `output[o] = sum_i concat_fp16[i] * W_O[o][i]`
- no bias is implemented

## Frozen Numeric Policy

- Hidden and weights are FP16.
- GEMV operands are exact FP16-to-FP32 extensions.
- GEMV uses the Stage 2 balanced FP32 reduction tree and tile accumulation
  order.
- Q/K/V projection outputs are FP32, then explicitly quantized to FP16.
- Stage 5 per-head outputs are FP32.
- The bit model keeps logical `head_output_fp32`, `concat_fp32`,
  `concat_fp16`, and `wo_output_fp32` trace nodes.
- RTL streams Stage 5 FP32 output lanes through one `fp32_to_fp16` converter and
  stores only the FP16 concat vector for W_O.
- FP32-to-FP16 policy is RNE, FTZ for FP32/FP16 subnormals, finite saturation on
  overflow, signed-zero preserving, and invalid for NaN/Inf.

## RTL Structure

Final top:

- `rtl/attention/projection_integrated_mha.sv`

Stage 6E support modules:

- `rtl/projection/head_concat_quantizer.sv`
- `rtl/projection/concat_fp16_buffer.sv`
- `rtl/projection/output_projection_controller.sv`

Projection datapath:

- One `projection_controller`
- One `shared_gemv_projection_core`
- One underlying `reconfigurable_pe_core`

Q, K, V, and W_O all reuse this single projection GEMV datapath. No third PE or
new W_O-specific floating multiply-add array was added. Stage 5 attention
internals are not restructured.

## Token Transaction Order

For a successful token:

1. hidden dimensions load;
2. Q projection;
3. K projection;
4. V projection;
5. Q/K/V FP16 stream to Stage 5;
6. Stage 5 attention completes;
7. Stage 5 atomically commits all-head K/V;
8. concat quantization completes;
9. W_O output projection completes;
10. final tiled FP32 output is emitted;
11. final done handshakes;
12. only then can the next hidden token start.

If QKV projection or Stage 5 fails before commit, Stage 5 abort semantics are
preserved, `valid_seq_len` does not increase, and W_O does not start. If Stage 5
already committed and concat/W_O later reports invalid, the K/V commit is not
rolled back; final done reports invalid/status and reset clears top transaction
state.

## Final Top Interface

Weight load:

- `weight_valid`, `weight_ready`
- `weight_kind` (`WQ`, `WK`, `WV`, `WO`)
- `weight_output_index`, `weight_input_index`
- `weight_data_fp16`
- `weight_last`, `weight_commit`

Hidden input:

- `token_valid`, `token_ready`
- `token_dim`
- `token_hidden_fp16`
- `token_last_dim`
- `token_meta`

Final output:

- `output_valid`, `output_ready`
- `output_base_dim`
- `output_vector_fp32`
- `output_lane_mask`
- `output_status`
- `output_invalid`
- `output_meta`
- `output_last`

Done/state:

- `done_valid`, `done_ready`
- `done_status`
- `done_invalid`
- `done_meta`
- `done_valid_seq_len`
- `current_valid_seq_len`

Counters:

- `perf_generation_steps`
- `perf_total_cycles`
- `perf_q_projection_cycles`
- `perf_k_projection_cycles`
- `perf_v_projection_cycles`
- `perf_qkv_quantization_cycles`
- `perf_attention_cycles`
- `perf_concat_quantization_cycles`
- `perf_output_projection_cycles`
- `perf_projection_pe_stall_cycles`
- `perf_attention_pe_stall_cycles`
- `perf_sfu_stall_cycles`
- `perf_weight_stall_cycles`
- `perf_buffer_stall_cycles`
- `perf_output_stall_cycles`
- `perf_peak_valid_seq_len`

## Verification Performed

Host:

```bash
python scripts/sim/run_stage6a_tests.py
python scripts/sim/run_stage6b_tests.py
python scripts/sim/run_stage6c_tests.py
python scripts/sim/run_stage6d_tests.py
python scripts/sim/run_stage6e_tests.py
python scripts/sim/run_stage6_tests.py
```

Docker:

```bash
make stage6-test
make stage6-rtl-sim
make stage6-lint
make stage6-synth
make stage6e-rtl-sim
make stage6e-lint
make stage6e-synth
make stage6d-rtl-sim
make stage6d-lint
make stage6d-synth
make stage6c-rtl-sim
make stage6c-lint
make stage6c-synth
make stage6b-rtl-sim
make stage6b-lint
make stage6b-synth
make stage5-rtl-sim
make stage5-lint
make stage5-synth
```

Results:

- Stage 6A-6E Python/model/vector/py_compile: PASS
- Stage 6E final top VCS H1/D8, H2/D8, H4/D8, H2/D16: PASS
- Stage 6E lint/vlogan: PASS with no diagnostics
- Stage 6E DC analyze/elaborate/link/check_design: PASS
- Stage 6B/6C/6D and Stage 5 regressions after Stage 6E: PASS
- Final `stage6-*` unified commands: PASS

DC checks are structural analyze/elaborate/link/check_design only. No formal
area, power, frequency, WNS, STA, process timing, or layout result is produced.

## Known Limitations

- Scheduling is intentionally serial and throughput is provisional.
- Projection weights, concat storage, and K/V cache are behavioral memories for
  correctness closure and are not SRAM macros.
- D_MODEL=128 DC coverage is address/control/component elaboration, not a
  physical full-memory implementation claim.
- No complete Transformer layer is present; Norm/Residual/FFN are out of Stage 6
  scope.

## Next Action

Stage 6 is closed for projection-integrated MHA correctness. Do not start Stage
7 work in this thread.
