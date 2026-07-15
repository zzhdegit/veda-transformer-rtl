# Project State

## Current Stage

- Stage: Hardware Stage H9 final-closure progress checkpoint
- Status: HW-H9 IN PROGRESS, NOT ACCEPTED
- Branch: `hw/h9-sfu-pe-interleaving`
- Last update: 2026-07-15

Hardware Stage H9 adds a checkpoint implementation of paper Attention
full-array native mapping and SFU/PE element-serial interleaving infrastructure.
The checkpoint includes H9 Python reference/cycle models, bounded score and
probability stream buffers, an interleaved paper single-head RTL schedule path,
matched paper staged versus paper interleaved single-head RTL A/B baselines,
H9 Make targets, and H9 reports.

Hardware Stage H9 is not accepted yet. The Docker EDA environment `nailong`
successfully reran the H9 host/model, RTL, lint/vlogan, and DC structural
checks plus Stage5/6/7/8 regressions. The cycle model is calibrated to the
matched single-head RTL A/B counter interval for D_HEAD=8, 16, and 64 at seq
1/2/8/16/32/64. Multi-head H9 interleaved RTL, full-layer H9 interleaved RTL,
long-sequence/cache-full, H9 lint, H9 DC, and Stage5/6/7/8 regressions pass.
The remaining blockers are the full reset interrupt matrix, broad random
backpressure with at least 20 fixed seeds, and complete assertion execution
evidence with negative/bind proof. Do not write `HARDWARE STAGE H9 PASS` or
create an H9 accepted tag until those remaining HW-H9 exit conditions are
implemented, executed, and pass.

Stage 8 remains the accepted hardware baseline:

```text
HARDWARE STAGE H8 PASS
```

Stage 8 implements a Paper-Structured 8x8x2 reconfigurable PE Array and maps
the current Attention QK and sV paths onto it. The independent array, Python
bit-accurate model, RTL hierarchy, Attention adapter, A/B architecture
selection, counters, and reports are added and verified.

Stage 6 projection-integrated multi-head attention correctness remains accepted.
Stage 6 acceptance audit reset-coverage conditions are closed.

Stage 7 Pre-Norm Transformer layer correctness remains accepted at tag
`stage7-correctness-accepted`.

Throughput, physical memory, and timing pipeline remain provisional.

Stage 8 changes only Attention PE organization for QK and sV. Projection
WQ/WK/WV/WO and FFN W1/W2 remain on the legacy `reconfigurable_pe_core` paths.
Stage 8 does not implement SFU-PE interleaving, global array sharing, KV cache
eviction, LayerNorm, Post-Norm, GELU, SiLU, SwiGLU, bias, dropout, RoPE,
embedding, LM head, tokenizer, multiple layers, SRAM macro binding, STA,
layout, or PPA.

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
- Stage 6E final top directed reset scenarios: PASS for H1/D8, H2/D8, H4/D8,
  and H2/D16. Covered reset during Q, K, V, QKV stream, attention, concat
  quantization, W_O, final output stall, and final done stall, followed by
  clean one-token recovery after weight reload.
- Host `python scripts/sim/run_stage5_tests.py`: PASS for 31 Python/model tests;
  host VCS was unavailable and skipped by the script.
- Docker audit closure bundle `make stage5-test stage5-rtl-sim stage5-lint
  stage5-synth stage6-test stage6-rtl-sim stage6-lint stage6-synth`: PASS.

DC checks are structural analyze/elaborate/link/check_design only. No formal
area, power, frequency, WNS, STA, process timing, or layout result is produced.

## Known Limitations

- Scheduling is intentionally serial and throughput is provisional.
- Projection weights, concat storage, and K/V cache are behavioral memories for
  correctness closure and are not SRAM macros.
- D_MODEL=128 DC coverage is address/control/component elaboration, not a
  physical full-memory implementation claim.
- Stage 6 alone remains projection-integrated MHA only; the complete single
  Pre-Norm Transformer layer is the Stage 7 top.

## Stage 8 Scope

Stage 8 replaces only the Stage 5/6 Attention PE path used for:

```text
Q x K^T
P x V
```

The new array is paper-structured but uses the repository-frozen arithmetic
contract:

- 8 rows x 8 columns x 2 groups = 128 physical PE cells.
- Type-A and Type-B cells are instantiated explicitly in the hierarchy.
- FP16 operands expand exactly to FP32.
- Products, partial sums, local accumulators, and outputs remain FP32 at the
  same boundaries used by Stage 5/6/7.
- QK uses `MODE_INNER_PRODUCT`.
- sV uses `MODE_OUTER_PRODUCT`.
- Softmax remains the existing staged serial SFU path between QK and sV.
- K/V cache layout, current-token semantics, all-head atomic commit, and
  cache-full behavior are unchanged.

Architecture selection:

- `ATTENTION_PE_ARCH=0`: legacy `reconfigurable_pe_core` Attention PE path.
- `ATTENTION_PE_ARCH=1`: paper array Attention adapter path.

Only the selected generate branch is instantiated in a given elaboration.

Stage 8 added:

- `docs/stage_08/paper_evidence.md`
- `docs/stage_08/spec.md`
- `docs/stage_08/mapping.md`
- `docs/stage_08/legacy_comparison.md`
- `model/pe_array/`
- `model/attention/paper_attention_reference.py`
- `model/attention/paper_attention_cycle_model.py`
- `rtl/pe/paper/`
- `rtl/attention/paper/paper_attention_adapter.sv`
- Stage 8 model and RTL testbenches under `tb/model/` and `tb/rtl/stage8/`
- Stage 8 simulation, lint, and synthesis scripts.
- Stage 8 Makefile targets `stage8-test`, `stage8-rtl-sim`, `stage8-lint`,
  and `stage8-synth`.

Stage 8 verification:

```bash
python scripts/sim/run_stage8_tests.py
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage8-test'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage8-rtl-sim'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage8-lint'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage8-synth'
```

Results:

- Paper evidence/spec checks: PASS.
- Stage 8 bit-accurate 8x8x2 PE array model tests: PASS.
- Independent array RTL VCS tests for PE cell, group, inner mode, outer mode,
  mode switch, reset, and backpressure: PASS.
- Attention paper-path RTL VCS tests for H1/D8, H2/D8, H4/D8, and H2/D16:
  PASS.
- Full `transformer_layer` RTL VCS tests with `ATTENTION_PE_ARCH=1` for H1/D8,
  H2/D8, H2/D8 two-token, H4/D8, and H2/D16: PASS.
- Stage 8 lint/vlogan: PASS with only accepted DesignWare pragma-no-effect
  warnings.
- Stage 8 DC analyze/elaborate/link/check_design structural checks: PASS for
  legacy and paper configurations; hierarchy reports count 128
  `paper_pe_cell` instances in paper-path tops.
- Stage 5/6/7 regressions after Stage 8 changes: PASS.

Stage 8 known limitations:

- The Stage 8D adapter is correctness-first and maps current `PE_NUM=8` tiles
  into the low paper-array lanes; it does not exploit full 128-cell throughput.
- Dense Attention RTL coverage is provided through the paper Attention tests;
  full-layer vectors continue to use the existing Stage 7 wrapper coverage.
- Full-layer RTL does not expose every internal PE partial-sum node.
- DC is structural only and is not an area, power, timing, or frequency result.
- SFU-PE interleaving, global array sharing, physical SRAM replacement, cache
  eviction, and PPA remain future work.

## Stage 7A Scope

Stage 7 implements one decoder-style Pre-Norm Transformer layer around the
accepted Stage 6 projection-integrated MHA top:

```text
n1 = RMSNorm(x)
a  = MHA(n1)
r1 = x + a
n2 = RMSNorm(r1)
h1 = W1(n2)
h  = ReLU(h1)
f  = W2(h)
y  = r1 + f
```

Frozen Stage 7 relations and first implementation choices:

- `D_MODEL = N_HEAD * D_HEAD`
- `D_FFN = 4 * D_MODEL`
- `D_MODEL` is power-of-two for RMSNorm mean scaling.
- RMSNorm input and residual paths are FP32.
- RMSNorm gamma, FFN weights, MHA input, and FFN activation storage are FP16.
- RMSNorm sum of squares is dimension-order sequential fused MAC.
- RMSNorm mean scale is an exact FP32 power-of-two constant.
- `EPS_FP32 = 32'h3727_C5AC`.
- RMSNorm apply order is `(x * inv_rms) * gamma`.
- Stage 7 instantiates exactly one frozen Stage 6 MHA instance.
- FFN is W1/ReLU/W2 without bias and uses output-row-major weights.
- ReLU maps negative finite values, signed zeros, NaN, and Inf to `+0`; NaN/Inf
  are invalid.

Stage 7 does not implement LayerNorm, Post-Norm, GELU, SiLU, SwiGLU, bias,
dropout, RoPE, embedding, LM head, tokenizer, multiple layers, SRAM macro
binding, STA, P&R, formal PPA, area, power, frequency, or WNS closure.

Stage 7A added:

- `docs/stage_07/spec.md`
- `model/transformer/rmsnorm_reference.py`
- `model/transformer/residual_reference.py`
- `model/transformer/relu_reference.py`
- `model/transformer/ffn_reference.py`
- `model/transformer/transformer_layer_reference.py`
- `model/transformer/transformer_layer_cycle_model.py`
- `tb/model/test_stage7_transformer_reference.py`
- `scripts/sim/run_stage7a_tests.py`
- `reports/stage_07/phase_7a_spec.md`
- `reports/stage_07/phase_7a_test_results.txt`

Stage 7B added:

- `rtl/arithmetic/fp32_sqrt_wrapper.sv`
- `rtl/transformer/rmsnorm_engine.sv`
- `rtl/transformer/residual_add_engine.sv`
- `tb/rtl/stage7/tb_stage7b_rmsnorm_residual.sv`
- `scripts/sim/gen_stage7b_vectors.py`
- `scripts/sim/run_stage7b_vcs.sh`
- `scripts/lint/run_stage7b_lint.py`
- `scripts/synth/run_stage7b_synth_check.py`
- `scripts/synth/stage7b_elaborate.tcl`
- Stage 7B Makefile targets: `stage7b-test`, `stage7b-rtl-sim`,
  `stage7b-lint`, and `stage7b-synth`.

Stage 7C added:

- `rtl/transformer/ffn_engine.sv`
- `tb/rtl/stage7/tb_stage7c_ffn_engine.sv`
- `scripts/sim/gen_stage7c_vectors.py`
- `scripts/sim/run_stage7c_vcs.sh`
- `scripts/lint/run_stage7c_lint.py`
- `scripts/synth/run_stage7c_synth_check.py`
- `scripts/synth/stage7c_elaborate.tcl`
- Stage 7C Makefile targets: `stage7c-test`, `stage7c-rtl-sim`,
  `stage7c-lint`, and `stage7c-synth`.

Stage 7D added:

- `rtl/transformer/transformer_layer.sv`
- `tb/rtl/stage7/tb_stage7d_transformer_layer.sv`
- `scripts/sim/gen_stage7d_vectors.py`
- `scripts/sim/run_stage7d_vcs.sh`
- `scripts/lint/run_stage7d_lint.py`
- `scripts/synth/run_stage7d_synth_check.py`
- `scripts/synth/stage7d_elaborate.tcl`
- `reports/stage_07/phase_7d_summary.md`
- Stage 7D Makefile targets: `stage7d-test`, `stage7d-rtl-sim`,
  `stage7d-lint`, and `stage7d-synth`.

Stage 7A verification:

```bash
python scripts/sim/run_stage7a_tests.py
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7a-test'
```

Results:

- Stage 7A model regression: PASS.
- Stage 7 transformer reference tests plus Stage 6E model tests: PASS.
- Python compile sweep: PASS.
- Stage 7 no-stall cycle model examples: PASS.
- Host `make stage7a-test` was not available because `make` is not installed on
  the Windows host; the Docker `make stage7a-test` flow passed.

Stage 7B verification:

```bash
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7b-test'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7b-rtl-sim'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7b-lint'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7b-synth'
```

Results:

- Stage 7B vectors for D_MODEL 8 and 16: PASS.
- Stage 7B RMSNorm/residual RTL VCS simulations for D_MODEL 8 and 16: PASS.
- Stage 7B lint/vlogan: PASS with only DesignWare pragma-no-effect warnings.
- Stage 7B DC analyze/elaborate/link/check_design: PASS for `fp32_sqrt_wrapper`,
  `rmsnorm_engine`, and `residual_add_engine`, including D_MODEL 128 structural
  elaboration.

Stage 7C verification:

```bash
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7c-test'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7c-rtl-sim'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7c-lint'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7c-synth'
```

Results:

- Stage 7C vectors for D_MODEL 8 and 16: PASS.
- Stage 7C FFN/ReLU RTL VCS simulations for D_MODEL 8 and 16: PASS.
- Stage 7C lint/vlogan: PASS with no diagnostics.
- Stage 7C DC analyze/elaborate/link/check_design: PASS for `ffn_engine`
  D_MODEL 8 and 16.

Stage 7D verification:

```bash
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7d-test'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7d-rtl-sim'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7d-lint'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7d-synth'
```

Results:

- Stage 7D vectors for H1/D8, H2/D8, H4/D8, H2/D16, and H2/D8 two-token:
  PASS.
- Stage 7D full `transformer_layer` RTL VCS simulations for H1/D8, H2/D8,
  H4/D8, and H2/D16 single-token vectors: PASS.
- Stage 7D H2/D8 two-token full-layer VCS sequence test: PASS.
- Stage 7D lint/vlogan: PASS with only DesignWare pragma-no-effect warnings.
- Stage 7D DC analyze/elaborate/link/check_design: PASS for
  `transformer_layer` H1/D8, H2/D8, H4/D8, and H2/D16.
- Stage 7D top-level output and done ready/valid backpressure are covered by
  directed RTL simulation.

## Next Action

Continue Hardware Stage H9 only after closing the open acceptance items recorded
in `reports/hw_h9/acceptance_audit.md`. Multi-head, full-layer,
long-sequence/cache-full, cycle calibration, lint, DC, and Stage5/6/7/8
regressions have been executed and pass. The remaining H9 work is to implement
and run the full reset interrupt matrix, broad random backpressure with at
least 20 fixed seeds, and complete assertion execution evidence without
changing the frozen Stage 5/6/7 numeric and transaction contracts.

Do not enter Hardware Stage H10, write `HARDWARE STAGE H9 PASS`, or create an
H9 accepted tag until all HW-H9 exit conditions are closed. Do not claim SRAM
macro binding, STA, P&R, area, power, frequency, WNS, or PPA until a future
stage adds the required technology libraries, memory macros, constraints,
layout, and reports.
