# Stage Handoff

## Stage

Stage 7A: Pre-Norm Transformer Layer Specification and Bit Model

## Status

STAGE 7A PASS. Stage 7 RTL implementation in progress.

Pre-Norm Transformer layer specification and Python bit-model framework are
frozen for implementation. Stage 7 RTL is not yet accepted.

Stage 6 projection-integrated multi-head attention correctness remains accepted,
and the Stage 6 acceptance audit is closed.

throughput, physical memory, and timing pipeline provisional.

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

Final top:

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

## Not Completed

- RMSNorm, LayerNorm, residual paths.
- FFN, GELU, SiLU, SwiGLU.
- Complete Transformer layer.
- SRAM macro binding or physical memory replacement.
- Timing pipeline closure.
- STA, P&R, formal PPA, area, power, frequency, WNS, or layout.
- Stage 7 RTL modules, RTL simulation, lint/vlogan, and DC structural checks.

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

## Transaction Semantics

Successful token order:

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

## Verification Results

Stage 7A:

- `python scripts/sim/run_stage7a_tests.py`: PASS
- `docker exec nailong bash -lc 'cd /workspace/VEDA && make stage7a-test'`: PASS

Host `make stage7a-test` was not available because `make` is not installed on
the Windows host. The underlying Python runner passed on host, and Docker make
passed in the Linux verification environment.

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

- Preserve Stage 5 all-head atomic commit and current-token causal semantics.
- Do not duplicate the shared projection PE datapath unless an accepted future
  spec explicitly changes the resource-sharing rule.
- Keep behavioral memories out of PPA claims.
- Stage 6 does not complete a Transformer layer; Norm/Residual/FFN remain out of
  scope.
- Stage 7 implementation must follow `docs/stage_07/spec.md` rather than the
  legacy full-layer planning text in `transformer_rtl_plan_md/06_full_transformer_layer.md`.
- Stage 7 must not claim PASS until RTL simulation, lint/vlogan, and DC
  structural checks are added and passing.
- Stage 7 must preserve the frozen Stage 6 child interface and commit semantics.
