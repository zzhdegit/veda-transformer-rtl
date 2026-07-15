# Stage Handoff

## Stage

Hardware Stage H9 undergraduate-thesis accepted baseline: Full-Array Attention
Mapping and SFU-PE Element-Serial Interleaving

## Status

HARDWARE STAGE H9 PASS — UNDERGRADUATE THESIS SCOPE.

Strict status: STRICT IP-GRADE H9 VERIFICATION NOT CLOSED.

Hardware Stage H9 has an undergraduate-thesis accepted implementation and
verification package for paper-native Attention mapping plus SFU/PE stream
interleaving infrastructure. The accepted thesis baseline passes H9 host/model
tests, matched single-head paper staged versus paper interleaved RTL A/B, H9
multi-head RTL, H9 full-layer RTL, long-sequence/cache-full RTL, H9
lint/vlogan, H9 DC structural checks, direct reset/random stress, independent
multi-head reset/random stress, assertion positive/negative execution, and
Stage5/6/7/8 regressions in the Docker EDA environment `nailong`.

Accepted baseline commit: the commit pointed to by
`hw-h9-sfu-pe-interleaving-thesis-accepted`.
Accepted baseline tag: `hw-h9-sfu-pe-interleaving-thesis-accepted`.

The strict IP-grade target is not closed. Full internal `transformer_layer`
reset injection and broad full-layer internal multi-endpoint random
backpressure remain deferred verification enhancements, not undergraduate
thesis blockers. Do not claim full IP-grade verification closure from this H9
baseline. Hardware Stage H10 has not started.

Hardware Stage H9 is now the undergraduate-thesis accepted hardware baseline.
Stage 8 remains the previous accepted paper-array correctness baseline.

The Stage 8 implementation adds a repository bit-accurate paper-array model,
an explicit 8 row x 8 column x 2 group RTL hierarchy with 128 PE cells, and a
paper Attention adapter selected by `ATTENTION_PE_ARCH=1`.

Stage 6 projection-integrated multi-head attention correctness remains accepted,
and the Stage 6 acceptance audit is closed.

Stage 7 Pre-Norm Transformer layer correctness remains accepted at
`stage7-correctness-accepted`.

Throughput, physical memory, and timing pipeline remain provisional.

## Completed

- Stage 6A numeric and interface boundary freeze.
- Stage 6B FP32-to-FP16 converter and shared projection GEMV path.
- Stage 6C Q/K/V projection engine and QKV staging.
- Stage 6D QKV projection integration with the accepted Stage 5
  multi-head generation engine.
- Stage 6E streamed head concat, FP32-to-FP16 concat quantization, FP16 concat
  storage, W_O projection, and final top.
- Stage 6F final unified commands, end-to-end verification, documentation, and
  closure.
- Stage 6 acceptance audit reset-coverage closure: final-top directed reset
  scenarios now interrupt Q, K, V, QKV stream, attention, concat quantization,
  W_O, final output stall, and final done stall, then verify clean one-token
  recovery after weight reload.
- Stage 7A Pre-Norm Transformer layer contract frozen in
  `docs/stage_07/spec.md`.
- Stage 7A Python bit-model framework added for RMSNorm, residual add, ReLU,
  FFN, full one-layer composition, and cycle estimates.
- Stage 7A model tests added for RMSNorm numeric boundaries, residual/ReLU,
  FFN layout, integrated Stage 6 MHA reuse, multi-token behavior, and cache-full
  semantics through the Stage 7 wrapper.
- Stage 7B `fp32_sqrt_wrapper` added as the required RMSNorm square-root
  arithmetic wrapper.
- Stage 7B `rmsnorm_engine` added with serial dimension-order sum-square fused
  MAC, exact power-of-two mean scaling, `EPS_FP32`, sqrt, reciprocal, frozen
  `(x * inv_rms) * gamma` apply order, and FP32-to-FP16 output quantization.
- Stage 7B `residual_add_engine` added with serial FP32 add-wrapper residual
  output.
- Stage 7B RTL testbench and scripts added for D_MODEL 8 and 16 RMSNorm and
  residual bit checks with output backpressure.
- Stage 7C `ffn_engine` added with one shared `reconfigurable_pe_core` for W1
  and W2, ReLU clamp/invalid handling, FP32-to-FP16 activation quantization,
  and final FP32 W2 outputs.
- Stage 7C RTL testbench and scripts added for D_MODEL 8 and 16 FFN checks with
  output backpressure.
- Stage 7D `transformer_layer` added around exactly one frozen Stage 6
  `projection_integrated_mha` child, two `rmsnorm_engine` instances, two
  `residual_add_engine` instances, and one `ffn_engine` instance.
- Stage 7D top integrates the frozen Pre-Norm order:
  RMSNorm1, MHA, residual1, RMSNorm2, FFN/ReLU, residual2, final tiled FP32
  output, and layer done.
- Stage 7D RTL testbench and scripts added for H1/D8, H2/D8, H4/D8, H2/D16,
  plus an H2/D8 two-token sequence test that checks valid sequence length 1
  then 2 through the full wrapper.
- Stage 8A paper evidence table and paper-structured PE array specification
  frozen in `docs/stage_08/`.
- Stage 8B bit-accurate 8x8x2 PE array Python model added with explicit rows,
  columns, groups, PE types, masks, reductions, mode switch, reset, and
  repeated-command coverage.
- Stage 8C independent RTL array added under `rtl/pe/paper/`, including PE
  cells, Type-A/Type-B wrappers, group hierarchy, L1/L2 reduction, controller,
  counters, assertions, and hierarchy audit reports.
- Stage 8D Attention integration added under `rtl/attention/paper/`, mapping
  QK to inner-product mode and sV to outer-product mode while preserving the
  legacy Attention path and selecting the implementation with
  `ATTENTION_PE_ARCH`.
- Stage 8D full-layer integration added for `transformer_layer` with
  `ATTENTION_PE_ARCH=LEGACY_PE` and `ATTENTION_PE_ARCH=PAPER_ARRAY`.

Current complete-layer top:

- `rtl/transformer/transformer_layer.sv`

Current Attention top:

- `rtl/attention/projection_integrated_mha.sv`

Stage 6E modules:

- `rtl/projection/head_concat_quantizer.sv`
- `rtl/projection/concat_fp16_buffer.sv`
- `rtl/projection/output_projection_controller.sv`

Model and test additions:

- `model/projection/projection_mha_reference.py`
- `model/projection/projection_mha_cycle_model.py`
- `tb/model/test_stage6e_output_projection.py`
- `tb/rtl/stage6/tb_projection_integrated_mha_stage6e.sv`
- `scripts/sim/gen_stage6e_vectors.py`
- `scripts/sim/run_stage6e_tests.py`
- `scripts/sim/run_stage6e_vcs.sh`
- `scripts/sim/run_stage6_tests.py`
- `scripts/sim/run_stage6_vcs.sh`
- `scripts/lint/run_stage6e_lint.py`
- `scripts/lint/run_stage6_lint.py`
- `scripts/synth/run_stage6e_synth_check.py`
- `scripts/synth/run_stage6_synth_check.py`
- `scripts/synth/stage6e_elaborate.tcl`

Audit closure also updated:

- `reports/stage_06/acceptance_audit.md`
- `PROJECT_STATE.md`
- `HANDOFF.md`

Stage 7A additions:

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
- `reports/stage_07/summary.md`

Stage 7B additions:

- `rtl/arithmetic/fp32_sqrt_wrapper.sv`
- `rtl/transformer/rmsnorm_engine.sv`
- `rtl/transformer/residual_add_engine.sv`
- `tb/rtl/stage7/tb_stage7b_rmsnorm_residual.sv`
- `scripts/sim/gen_stage7b_vectors.py`
- `scripts/sim/run_stage7b_vcs.sh`
- `scripts/lint/run_stage7b_lint.py`
- `scripts/synth/run_stage7b_synth_check.py`
- `scripts/synth/stage7b_elaborate.tcl`
- `reports/stage_07/phase_7b_vcs_rtl_sim.txt`
- `reports/stage_07/phase_7b_lint_results.txt`
- `reports/stage_07/phase_7b_synth_check.txt`

Stage 7C additions:

- `rtl/transformer/ffn_engine.sv`
- `tb/rtl/stage7/tb_stage7c_ffn_engine.sv`
- `scripts/sim/gen_stage7c_vectors.py`
- `scripts/sim/run_stage7c_vcs.sh`
- `scripts/lint/run_stage7c_lint.py`
- `scripts/synth/run_stage7c_synth_check.py`
- `scripts/synth/stage7c_elaborate.tcl`
- `reports/stage_07/phase_7c_vcs_rtl_sim.txt`
- `reports/stage_07/phase_7c_lint_results.txt`
- `reports/stage_07/phase_7c_synth_check.txt`

Stage 7D additions:

- `rtl/transformer/transformer_layer.sv`
- `tb/rtl/stage7/tb_stage7d_transformer_layer.sv`
- `scripts/sim/gen_stage7d_vectors.py`
- `scripts/sim/run_stage7d_vcs.sh`
- `scripts/lint/run_stage7d_lint.py`
- `scripts/synth/run_stage7d_synth_check.py`
- `scripts/synth/stage7d_elaborate.tcl`
- `reports/stage_07/phase_7d_summary.md`
- `reports/stage_07/phase_7d_vcs_rtl_sim.txt`
- `reports/stage_07/phase_7d_lint_results.txt`
- `reports/stage_07/phase_7d_synth_check.txt`

Stage 8 additions:

- `docs/stage_08/paper_evidence.md`
- `docs/stage_08/spec.md`
- `docs/stage_08/mapping.md`
- `docs/stage_08/legacy_comparison.md`
- `model/pe_array/paper_pe_reference.py`
- `model/pe_array/paper_pe_group_reference.py`
- `model/pe_array/paper_array_8x8x2_reference.py`
- `model/pe_array/paper_array_mapping.py`
- `model/pe_array/paper_array_cycle_model.py`
- `model/pe_array/paper_array_compare_legacy.py`
- `model/attention/paper_attention_reference.py`
- `model/attention/paper_attention_cycle_model.py`
- `rtl/pe/paper/`
- `rtl/attention/paper/paper_attention_adapter.sv`
- `tb/model/test_stage8_paper_array.py`
- `tb/model/test_stage8_paper_attention.py`
- `tb/rtl/stage8/`
- `scripts/sim/run_stage8_tests.py`
- `scripts/sim/run_stage8b_tests.py`
- `scripts/sim/run_stage8c_tests.py`
- `scripts/sim/run_stage8d_tests.py`
- `scripts/sim/run_stage8c_vcs.sh`
- `scripts/sim/run_stage8d_vcs.sh`
- `scripts/sim/run_stage8d_layer_vcs.sh`
- `scripts/lint/run_stage8c_lint.py`
- `scripts/lint/run_stage8d_lint.py`
- `scripts/synth/run_stage8c_synth_check.py`
- `scripts/synth/run_stage8d_synth_check.py`
- `scripts/synth/stage8c_elaborate.tcl`
- `scripts/synth/stage8d_elaborate.tcl`
- `reports/stage_08/`

Hardware Stage H9 checkpoint additions:

- `docs/hw_h9/paper_schedule_evidence.md`
- `docs/hw_h9/spec.md`
- `docs/hw_h9/stream_protocol.md`
- `docs/hw_h9/full_array_mapping.md`
- `docs/hw_h9/softmax_schedule.md`
- `docs/hw_h9/verification_plan.md`
- `docs/hw_h9/thesis_acceptance_scope.md`
- `model/attention/paper_interleaved_attention_reference.py`
- `model/attention/paper_interleaved_softmax_reference.py`
- `model/attention/paper_interleaved_cycle_model.py`
- `model/attention/paper_interleaved_compare_h8.py`
- H9 native mapping helpers in `model/pe_array/paper_array_mapping.py`
- H9 D_HEAD scale constants in `model/attention/single_head_reference.py`
  and `rtl/attention/attention_score_scaler.sv`
- `rtl/attention/paper/interleaved/`
- `ATTENTION_SCHEDULE` selection propagated through single-head, multi-head,
  projection-integrated MHA, and transformer-layer tops.
- `tb/model/test_hw_h9_interleaved_attention.py`
- `tb/rtl/hw_h9/`
- `scripts/sim/run_hw_h9_tests.py`
- `scripts/sim/run_hw_h9_vcs.sh`
- `scripts/sim/run_hw_h9_thesis_acceptance.sh`
- `scripts/lint/run_hw_h9_lint.py`
- `scripts/synth/run_hw_h9_synth_check.py`
- `scripts/synth/hw_h9_elaborate.tcl`
- Make targets `hw-h9-test`, `hw-h9-model-test`, `hw-h9-buffer-test`,
  `hw-h9-overlap-test`, `hw-h9-ab-compare`, `hw-h9-rtl-sim`,
  `hw-h9-lint`, `hw-h9-synth`, and `hw-h9-thesis-acceptance`.
- `reports/hw_h9/`

## Not Completed

- LayerNorm.
- Post-Norm, GELU, SiLU, SwiGLU, bias, dropout, RoPE, embedding, LM head,
  tokenizer, and multiple layers.
- SRAM macro binding or physical memory replacement.
- Timing pipeline closure.
- STA, P&R, formal PPA, area, power, frequency, WNS, or layout.
- Strict IP-grade full-layer internal reset injection at every requested
  `transformer_layer` micro-stage.
- Strict IP-grade broad full-layer internal multi-endpoint random
  backpressure.
- Functional coverage closure, assertion coverage closure, and formal property
  proof.
- KV cache eviction.
- Global array sharing across Projection, Attention, and FFN.
- Paper-exact arithmetic claims beyond the evidence in
  `docs/stage_08/paper_evidence.md`.

## Architecture Notes

Q, K, V, and W_O share one `projection_controller`, one
`shared_gemv_projection_core`, and one underlying `reconfigurable_pe_core`.
There is no extra W_O PE array.

The final top does not instantiate `qkv_projection_engine`; that engine remains
for Stage 6C/6D regression compatibility. Stage 5 attention internals remain the
accepted Stage 5 implementation.

Concat is logical FP32 in the bit model and physical FP16 in RTL:

```text
concat_index = output_head * D_HEAD + output_base_dim + lane_index
```

RTL streams active Stage 5 FP32 lanes through one `fp32_to_fp16` converter and
writes only `concat_fp16[D_MODEL]`.

W_O uses:

```text
output[o] = sum(i = 0..D_MODEL-1) concat_fp16[i] * W_O[o][i]
```

with output-row-major `W_O[output_index][input_index]`, no bias, and the Stage 2
balanced reduction/tile accumulation order.

Stage 7 final top is `rtl/transformer/transformer_layer.sv`. It instantiates
exactly one frozen `projection_integrated_mha` child, two `rmsnorm_engine`
instances, two `residual_add_engine` instances, and one `ffn_engine` instance.
The FFN engine retains exactly one `reconfigurable_pe_core` for W1 and W2.

Stage 7 external weight kinds are WQ, WK, WV, WO, NORM1_GAMMA, NORM2_GAMMA,
FFN_W1, and FFN_W2. WQ/WK/WV/WO route to the Stage 6 child; gamma and FFN
weights route to the Stage 7 engines.

Stage 8 keeps Projection WQ/WK/WV/WO and FFN W1/W2 on the legacy
`reconfigurable_pe_core` paths. Only the Attention QK and sV PE path changes
when `ATTENTION_PE_ARCH=1`.

The paper array hierarchy is:

```text
8 rows x 8 columns x 2 groups = 128 physical PE cells
```

QK maps to `MODE_INNER_PRODUCT`. sV maps to `MODE_OUTER_PRODUCT`. The adapter
is correctness-first: current `PE_NUM=8` tiles are placed in the low
paper-array lanes and tail masks keep inactive cells from accumulating. Softmax
remains the existing staged SFU path, so Stage 8 still executes:

```text
QK complete
-> existing Softmax/SFU
-> sV
```

The legacy path remains selectable with `ATTENTION_PE_ARCH=0`. A given
elaboration instantiates only the selected generate branch.

Hardware Stage H9 checkpoint keeps the Stage 8 staged paper path available and
adds `ATTENTION_SCHEDULE`:

```text
ATTENTION_SCHEDULE=0: staged schedule
ATTENTION_SCHEDULE=1: interleaved paper schedule
```

Legal checkpoint combinations are legacy staged, paper staged, and paper
interleaved. Legacy interleaved is rejected. The H9 native dimension mapping is:

```text
group  = dimension % 2
local  = dimension / 2
row    = local % 8
column = local / 8
cell   = group * 64 + row * 8 + column
```

D_HEAD=8 uses both groups and multiple rows; D_HEAD=16 uses both groups and
all rows; D_HEAD=64 uses both groups, all rows, and multiple columns; D_HEAD=128
is structurally covered by the model. The interleaved checkpoint uses bounded
score/probability stream buffers and preserves the existing repository softmax
arithmetic rather than claiming a paper-exact bit-level SFU.

## Transaction Semantics

Successful Stage 6 token order:

```text
QKV projection
-> Stage 5 attention
-> Stage 5 all-head atomic K/V commit
-> concat quantization
-> W_O projection
-> final output
-> final done
-> next token may start
```

Cache-full and pre-commit Stage 5 errors do not increment `valid_seq_len` and do
not start W_O. Post-commit concat/W_O errors do not roll back K/V; final done
reports invalid/status.

Weight writes are blocked during an active token transaction. Reset clears top
transaction state.

Directed reset coverage verifies the final top clears reset-visible transaction
state, exposes no X on valid/status outputs, accepts a clean token after reset,
and does not duplicate the Stage 5 commit after recovery.

Successful Stage 7 token order:

```text
input load
-> RMSNorm1
-> Stage 6 MHA
-> residual1
-> RMSNorm2
-> FFN1
-> ReLU and activation quantization
-> FFN2
-> residual2
-> final FP32 output
-> layer done
-> next token may start
```

Stage 6 `done_valid` is internal MHA done. Stage 7 layer `done_valid` is emitted
only after residual2 output and final tiled FP32 output complete.

Stage 8 preserves K/V cache layout:

```text
K_cache[head][token][dimension]
V_cache[head][token][dimension]
```

Current-token attention, all-head atomic commit, cache-full behavior,
pre-commit abort behavior, post-commit invalid reporting, metadata propagation,
and valid sequence length semantics are unchanged.

## Verification Results

Hardware Stage H9 checkpoint:

- `python scripts/sim/run_hw_h9_tests.py`: PASS.
- `docker exec -w /workspace/VEDA nailong make hw-h9-test`: PASS.
- `docker exec -w /workspace/VEDA nailong make hw-h9-rtl-sim`: PASS.
- `docker exec -w /workspace/VEDA nailong make hw-h9-lint`: PASS.
- `docker exec -w /workspace/VEDA nailong make hw-h9-synth`: PASS.

H9 RTL simulation coverage completed:

- `tb_h9_score_buffer`: PASS.
- `tb_h9_probability_fifo`: PASS.
- `tb_h9_single_head` D_HEAD=8, 16, and 64: PASS.
- Single-head smoke counters show `qk_sfu_overlap=135`,
  `sfu_sv_overlap=66`, `group0=408`, and `group1=408` for each D_HEAD smoke
  run.
- Matched RTL A/B single-head `PAPER_ARRAY+STAGED` versus
  `PAPER_ARRAY+INTERLEAVED`: PASS for D_HEAD=8, 16, and 64, seq 1/2/8/16/32/64,
  with identical top, input data, DesignWare latency, and output/done ready
  environment.
- Matched deterministic output/done backpressure subset: PASS for D_HEAD=8, 16,
  and 64 at seq16 and seq32.

H9 lint/vlogan passed with only accepted DesignWare pragma-no-effect warnings.
H9 DC structural checks passed for legacy staged, paper staged, and paper
interleaved architecture/schedule selections. The H9 hierarchy reports count
128 `paper_pe_cell` occurrences in checked paper interleaved tops. DC results
remain analyze/elaborate/link/check_design only; no PPA is claimed.

The earlier structural H8/H9 cycle-model comparison was not apples-to-apples for
performance acceptance. The matched RTL A/B baseline is now the controlling
single-head performance evidence:

- D_HEAD=8 seq16/32: 1363/2707 staged versus 1169/2209 interleaved.
- D_HEAD=16 seq16/32: 2472/4920 staged versus 1171/2211 interleaved.
- D_HEAD=64 seq16/32: 9126/18198 staged versus 1183/2223 interleaved.

The cycle model is calibrated to exact matched RTL total cycles for D_HEAD 8,
16, and 64 at seq 1, 2, 8, 16, 32, and 64. The model's
`full_array_non_interleaved_cycles` and overlap subtotals remain structural
trend estimates, not acceptance counters.

Stage 8:

- `python scripts/sim/run_stage8_tests.py`: PASS
- `docker exec nailong bash -lc 'cd /workspace/VEDA && make stage8-test'`: PASS
- `docker exec nailong bash -lc 'cd /workspace/VEDA && make stage8-rtl-sim'`: PASS
- `docker exec nailong bash -lc 'cd /workspace/VEDA && make stage8-lint'`: PASS
- `docker exec nailong bash -lc 'cd /workspace/VEDA && make stage8-synth'`: PASS

Stage 8D Attention/full-layer VCS configurations:

- Paper single-head Attention D8 and D16.
- Paper multi-head Attention H1/D8, H2/D8, H4/D8, and H2/D16.
- Paper full `transformer_layer` H1/D8, H2/D8, H2/D8 two-token, H4/D8, and
  H2/D16.

Stage 8 lint/vlogan passed with only accepted DesignWare pragma-no-effect
warnings. Stage 8 DC structural checks include legacy and paper configurations
and hierarchy reports count exactly 128 `paper_pe_cell` instances in paper-path
tops.

Stage 7A:

- `python scripts/sim/run_stage7a_tests.py`: PASS
- `docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7a-test'`: PASS

Host `make stage7a-test` was not available because `make` is not installed on
the Windows host. The underlying Python runner passed on host, and Docker make
passed in the Linux verification environment.

Stage 7B:

- `docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7b-test'`: PASS
- `docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7b-rtl-sim'`: PASS
- `docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7b-lint'`: PASS
- `docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7b-synth'`: PASS

Stage 7B VCS configurations:

- RMSNorm/residual D_MODEL=8
- RMSNorm/residual D_MODEL=16

Stage 7B DC structural checks include `fp32_sqrt_wrapper`, D_MODEL 8/16/128
`rmsnorm_engine`, and D_MODEL 16/128 `residual_add_engine`.

Stage 7C:

- `docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7c-test'`: PASS
- `docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7c-rtl-sim'`: PASS
- `docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7c-lint'`: PASS
- `docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7c-synth'`: PASS

Stage 7C VCS configurations:

- FFN/ReLU D_MODEL=8, D_FFN=32
- FFN/ReLU D_MODEL=16, D_FFN=64

Stage 7C DC structural checks include D_MODEL 8/16 `ffn_engine`.

Stage 7D:

- `docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7d-test'`: PASS
- `docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7d-rtl-sim'`: PASS
- `docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7d-lint'`: PASS
- `docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7d-synth'`: PASS

Stage 7D VCS configurations:

- Full `transformer_layer` H1/D8, D_MODEL=8, one token
- Full `transformer_layer` H2/D8, D_MODEL=16, one token
- Full `transformer_layer` H2/D8, D_MODEL=16, two tokens
- Full `transformer_layer` H4/D8, D_MODEL=32, one token
- Full `transformer_layer` H2/D16, D_MODEL=32, one token

Stage 7D lint/vlogan passed with only DesignWare pragma-no-effect warnings.
Stage 7D DC structural checks include `transformer_layer` H1/D8, H2/D8,
H4/D8, and H2/D16.

Host:

- `python scripts/sim/run_stage6a_tests.py`: PASS
- `python scripts/sim/run_stage6b_tests.py`: PASS
- `python scripts/sim/run_stage6c_tests.py`: PASS
- `python scripts/sim/run_stage6d_tests.py`: PASS
- `python scripts/sim/run_stage6e_tests.py`: PASS
- `python scripts/sim/run_stage6_tests.py`: PASS
- `python scripts/sim/run_stage5_tests.py`: PASS; host VCS unavailable and
  skipped by the script

Docker:

- `make stage6-test`: PASS
- `make stage6-rtl-sim`: PASS
- `make stage6-lint`: PASS
- `make stage6-synth`: PASS
- `make stage6e-rtl-sim`: PASS
- `make stage6e-lint`: PASS
- `make stage6e-synth`: PASS
- `make stage6d-rtl-sim`: PASS
- `make stage6d-lint`: PASS
- `make stage6d-synth`: PASS
- `make stage6c-rtl-sim`: PASS
- `make stage6c-lint`: PASS
- `make stage6c-synth`: PASS
- `make stage6b-rtl-sim`: PASS
- `make stage6b-lint`: PASS
- `make stage6b-synth`: PASS
- `make stage5-rtl-sim`: PASS
- `make stage5-lint`: PASS
- `make stage5-synth`: PASS
- `make stage5-test stage5-rtl-sim stage5-lint stage5-synth stage6-test stage6-rtl-sim stage6-lint stage6-synth`: PASS

Final top VCS configurations:

- H1/D8
- H2/D8
- H4/D8
- H2/D16

The H2/D8 Stage 6E vector includes deterministic dense WQ/WK/WV/WO weights with
mixed signs, cancellation, powers-of-two values, and moderate magnitudes. All
final tiled FP32 outputs are bit-exact against the bit model. High precision is
used only for error statistics in the model tests and is not an RTL tolerance.

## Assertions

Stage 6 RTL/vlogan checks include assertion/stability tokens for:

- `d_model_equals_n_head_times_d_head`
- `weight_address_in_range`
- `all_required_weights_complete_before_start`
- `no_weight_write_while_active`
- `hidden_dimension_order_legal`
- `no_projection_without_complete_hidden`
- `qkv_head_dim_order_legal`
- `no_attention_before_qkv_complete`
- `current_token_causal_semantics_preserved`
- `valid_seq_len_changes_only_by_stage5_commit`
- `no_concat_before_head_output`
- `concat_index_in_range`
- `concat_write_only_for_active_lane`
- `no_duplicate_concat_write`
- `no_output_projection_before_concat_complete`
- `wo_uses_shared_projection_datapath`
- `no_next_token_before_final_done`
- `no_duplicate_final_output`
- `no_duplicate_final_done`
- `output_stable_until_ready`
- `done_stable_until_ready`
- `metadata_stable_until_ready`
- `no_unknown_output_when_valid`
- `transaction_count_conserved`
- `reset_clears_active_transaction`
- `cache_full_has_no_new_commit`
- `cache_commit_occurs_once_per_successful_token`

VCS runs compile assertions with `-assert svaext`.

## Dependencies

- Host Python 3.12 environment for model tests.
- Docker container `nailong` for VCS/vlogan/DC.
- Synopsys VCS/vlogan and Design Compiler inside the container.
- DesignWare simulation and foundation libraries inside the container.
- No PDK, standard-cell library, SRAM macro, license file, or EDA installation
  directory is committed to the repository.

## Reproduction Steps

From `D:\IC_Workspace\VEDA`:

```bash
python scripts/sim/run_stage8_tests.py
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage8-test'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage8-rtl-sim'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage8-lint'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage8-synth'
docker exec -w /workspace/VEDA nailong make hw-h9-test
docker exec -w /workspace/VEDA nailong make hw-h9-rtl-sim
docker exec -w /workspace/VEDA nailong make hw-h9-lint
docker exec -w /workspace/VEDA nailong make hw-h9-synth
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7d-test'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7d-rtl-sim'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7d-lint'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7d-synth'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7c-test'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7c-rtl-sim'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7c-lint'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7c-synth'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7b-test'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7b-rtl-sim'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7b-lint'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7b-synth'
python scripts/sim/run_stage7a_tests.py
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7a-test'
python scripts/sim/run_stage5_tests.py
python scripts/sim/run_stage6_tests.py
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage5-test stage5-rtl-sim stage5-lint stage5-synth stage6-test stage6-rtl-sim stage6-lint stage6-synth'
```

The narrower Stage 6 reproduction remains:

```bash
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage6-test'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage6-rtl-sim'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage6-lint'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage6-synth'
```

For phase-level regression:

```bash
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage6e-rtl-sim && make stage6e-lint && make stage6e-synth'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage6d-rtl-sim && make stage6d-lint && make stage6d-synth'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage6c-rtl-sim && make stage6c-lint && make stage6c-synth'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage6b-rtl-sim && make stage6b-lint && make stage6b-synth'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage5-rtl-sim && make stage5-lint && make stage5-synth'
```

## Next-Stage Cautions

- Hardware Stage H9 is accepted only for undergraduate thesis scope:
  `HARDWARE STAGE H9 PASS — UNDERGRADUATE THESIS SCOPE`.
- `STRICT IP-GRADE H9 VERIFICATION NOT CLOSED` remains true. Do not claim full
  IP-grade verification closure, functional coverage closure, assertion
  coverage closure, or formal proof.
- The H9 thesis baseline proves bounded-buffer single-head overlap, matched
  single-head RTL performance, implemented multi-head RTL, implemented
  full-layer RTL, deterministic cache-full coverage, direct reset/random
  stress, independent multi-head reset/random stress, and assertion
  positive/negative execution.
- Full internal `transformer_layer` reset injection and broad full-layer
  internal multi-endpoint random backpressure are deferred strict verification
  enhancements.
- Model Stage M3 may use the H9 thesis baseline only after a separate
  user-approved task. Hardware Stage H10 has not started.
- Do not claim H9 paper-exact SFU arithmetic. The checkpoint preserves the
  existing repository softmax arithmetic and marks missing paper details as
  repository design decisions.
- Do not claim QK and sV use the same paper array concurrently. The checkpoint
  only permits QK/SFU overlap and SFU/sV overlap after the safe inner-to-outer
  mode switch.
- Stage 8 remains closed for paper-structured array and Attention QK/sV mapping
  correctness.
- Preserve Stage 5 all-head atomic commit and current-token causal semantics.
- Do not duplicate the shared projection PE datapath unless an accepted future
  spec explicitly changes the resource-sharing rule.
- Keep behavioral memories out of PPA claims.
- Stage 6 alone does not complete a Transformer layer; the accepted complete
  single-layer top is Stage 7 `transformer_layer`.
- Future work must preserve `docs/stage_07/spec.md` and must not switch to the
  legacy full-layer planning text in
  `transformer_rtl_plan_md/06_full_transformer_layer.md`.
- Future changes must preserve the frozen Stage 6 child interface and commit
  semantics inside Stage 7.
- RMSNorm finite DW status bits such as inexact are diagnostic status, not
  `invalid`; invalid remains the hard error indicator.
- The Stage 7B/7C/7D engines are serial correctness engines, not
  throughput-optimized final scheduling.
- Stage 7D simulation uses Stage 6-style separate projection matrix commit
  pulses for WQ/WK/WV/WO, while Stage 7 RMSNorm/FFN foundation engines complete
  on their existing external commit markers.
- Future work must not claim SRAM macro binding, STA, P&R, area, power,
  frequency, WNS, or PPA until technology libraries, memory macros,
  constraints, layout, and reports are present.
