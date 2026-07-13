# Stage 6 Acceptance Audit

## 1. Audit Result

- Result: CONDITIONAL PASS
- Audit date: 2026-07-13
- Branch: `stage6-projection-mha`
- Final accepted commit: `c59ac45cd7bd59fbddaf486ea655b1d753342be9`
- Acceptance tag: `stage6-correctness-accepted`
- Tag target: `c59ac45cd7bd59fbddaf486ea655b1d753342be9`
- Remote sync: local `stage6-projection-mha` and `origin/stage6-projection-mha`
  matched before this audit report was added.

The functional, regression, tag, and restricted-file checks passed. The result
is conditional because the workspace had one pre-existing untracked file,
`transformer_rtl_plan_md/LATE_STAGE_REAL_MODEL_VALIDATION_PLAN.md`, so the
working tree was not completely clean. That file was not modified or staged by
this audit. Current final-top reset coverage is also limited to reset at test
start plus RTL reset assertions; directed reset-during-each-phase vectors were
not present in the accepted Stage 6E final-top testbench.

## 2. Accepted Functional Boundary

```text
hidden FP16
-> QKV projection
-> Q/K/V FP32-to-FP16 quantization
-> Stage 5 multi-head current-token causal attention
-> streamed head concat quantization
-> FP16 concat buffer
-> W_O
-> final FP32 output
```

Stage 6 does not include Norm, Residual, FFN, activation functions, SRAM macro
binding, timing closure, STA, layout, area, power, frequency, WNS, or a complete
Transformer layer.

Final conclusion remains:

```text
projection-integrated multi-head attention correctness accepted
throughput, physical memory, and timing pipeline provisional
```

## 3. Git And Tag Audit

| Check | Result |
| --- | --- |
| Repository root | `D:/IC_Workspace/VEDA` |
| Current branch | `stage6-projection-mha` |
| Remote URL | `https://github.com/zzhdegit/veda-transformer-rtl.git`; no credential in URL |
| Branch sync | Local branch matched `origin/stage6-projection-mha` at `c59ac45cd7bd59fbddaf486ea655b1d753342be9` before this report |
| Unknown ahead/behind | None observed before this report |
| Stage 6 tag | `stage6-correctness-accepted` exists locally and remotely |
| Stage 6 tag target | `c59ac45cd7bd59fbddaf486ea655b1d753342be9` |
| Stage 5 tag target | `stage5-correctness-accepted` peels to `c350c47c067e224285956d8a1bcd4a469abda672`; not moved |
| Worktree | Tracked files clean before this report; one untracked local plan file remained |

## 4. Restricted File Audit

Tracked path scan results:

- No tracked `build/`, `simv`, `csrc`, waveform, PDK, standard-cell library,
  SRAM macro, `.sldb`, `.db`, `.lib`, `.lef`, `.gds`, `.ndm`, or `.nlib` file
  was found.
- `tb/rtl/stage1b/tb_dw_fp_mac_semantics.sv` matched the `DW_fp_` path pattern
  by file name only. It is a repository testbench, not Synopsys DesignWare
  source.
- Text searches for token/credential/license-related words found policy and
  documentation references only. No secret-like token pattern was found.
- `LOCAL_GITHUB.md` is ignored by `.git/info/exclude`, is not tracked by Git,
  and no secret-like token pattern was found in it.

## 5. Acceptance Regression

| Command | Result | Notes |
| --- | --- | --- |
| `python scripts/sim/run_stage6_tests.py` | PASS | Host Python 3.12; 6A-6D report 16 tests each, 6E reports 18 tests; vector generation, py_compile, and cycle model passed |
| `docker exec nailong ... make stage6-test` | PASS | Docker fallback runner; 6A-6D 16 tests each, 6E 18 tests |
| `docker exec nailong ... make stage6-rtl-sim` | PASS | Stage 6B/6C/6D/6E VCS passed; final top configs H1/D8, H2/D8, H4/D8, H2/D16 |
| `docker exec nailong ... make stage6-lint` | PASS | Static hygiene passed; vlogan diagnostics: none |
| `docker exec nailong ... make stage6-synth` | PASS | DC analyze/elaborate/link/check_design passed for 6B/6C/6D/6E; no PPA generated |
| `docker exec nailong ... make stage5-rtl-sim` | PASS | Cache manager plus H1/D8, H2/D8, H4/D8, H2/D16 multi-head generation passed |
| `docker exec nailong ... make stage5-lint` | PASS | vlogan diagnostics: none |
| `docker exec nailong ... make stage5-synth` | PASS | DC analyze/elaborate/link/check_design passed |

Assertions are compiled in VCS through the existing `-assert svaext` flows.
Stage 6 lint reported `vlogan_diagnostics: none`. DC checks are structural only
and do not report area, power, frequency, WNS, STA, process timing, or layout.

## 6. Final Top Instance Audit

Static hierarchy tracing starts from `rtl/attention/projection_integrated_mha.sv`.
Old phase modules such as `qkv_projection_engine`, `projected_multi_head_attention`,
and `generation_attention_engine` remain in the repository for regression but
are not instantiated by the final Stage 6 top.

| Item | Final top actual count | Notes |
| --- | ---: | --- |
| `projection_controller` | 1 | `u_shared_projection_controller` |
| `shared_gemv_projection_core` | 1 | Instantiated inside the one projection controller |
| Projection `reconfigurable_pe_core` | 1 | Inside `shared_gemv_projection_core` |
| Stage 5 attention compute path | 1 | One `single_head_attention` through `multi_head_generation_engine` |
| Attention `reconfigurable_pe_core` | 1 | Inside the Stage 5 single-head attention controller |
| Total `reconfigurable_pe_core` | 2 | One projection PE path plus one attention PE path |
| `fp32_to_fp16` | 2 | One QKV quantizer plus one head concat quantizer |
| `fp32_mac_wrapper` | `2 * PE_NUM + 3` = 19 for PE_NUM=8 | Two PE cores contribute `2 * PE_NUM`; scaler, softmax reduction, and normalization add 3 |
| `fp32_add_wrapper` | `2 * PE_NUM + 2` = 18 for PE_NUM=8 | Two PE cores contribute reduction/tile adders; softmax reduction and normalization add 2 |
| `fp32_exp_wrapper` | 2 | Softmax reduction and normalization |
| `fp32_recip_wrapper` | 1 | Softmax normalization |

Q, K, V, and W_O therefore share one projection GEMV. W_O does not add an
independent PE or floating multiply-add array.

## 7. Numeric Coverage Audit

| Config | Directed weights | Dense deterministic WQ/WK/WV/WO | Multi token | Cache full | Backpressure | Reset |
| --- | --- | --- | --- | --- | --- | --- |
| H1/D8 | WQ/WK broadcast-column, WV identity, dense WO | No | Yes, MAX_SEQ_LEN+1 | Yes | Output and done | Initial reset only |
| H2/D8 | Dense deterministic WQ/WK/WV/WO | Yes | Yes, MAX_SEQ_LEN+1 | Yes | Output and done | Initial reset only |
| H4/D8 | WQ/WK broadcast-column, WV identity, dense WO | No | Yes, MAX_SEQ_LEN+1 | Yes | Output and done | Initial reset only |
| H2/D16 | WQ/WK broadcast-column, WV identity, dense WO | No | Yes, MAX_SEQ_LEN+1 | Yes | Output and done | Initial reset only |

Coverage notes:

- Dense deterministic WQ/WK/WV/WO is currently final-top RTL coverage for H2/D8.
- H1/D8, H4/D8, and H2/D16 use directed sparse/exact QKV matrices and dense W_O.
- The Stage 6 Python model tests compare trace nodes including hidden FP16,
  Q/K/V projection and quantization, logical concat FP32, concat FP16, W_O
  FP32, and final output FP32.
- The Stage 6E final-top RTL testbench compares final tiled FP32 outputs,
  output metadata/status, done metadata/status, and valid sequence length
  against generated bit-model vectors. It does not independently sample every
  internal RTL node in the final top.
- High-precision output projection is diagnostic only. It is used for error
  statistics such as max absolute error, MAE, RMSE, relative L2, and cosine
  similarity in model tests; it is not used as an RTL tolerance.
- Reset-during-Q/K/V/QKV-stream/attention/concat/W_O/final-output-stall remains
  a documented coverage gap for this accepted Stage 6 test set.

## 8. Cycle Baseline

No-stall cycle model, single token with `seq_len_before=0`:

| Config | Hidden load | Q | K | V | QKV quant | Attention | Concat quant | W_O | Final output | Control overhead | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H1/D8 | 8 | 24 | 24 | 24 | 24 | 30 | 8 | 32 | 1 | 13 | 188 |
| H2/D8 | 16 | 64 | 64 | 64 | 48 | 60 | 16 | 80 | 2 | 14 | 428 |
| H4/D8 | 32 | 192 | 192 | 192 | 96 | 120 | 32 | 224 | 4 | 16 | 1100 |
| H2/D16 | 32 | 192 | 192 | 192 | 96 | 120 | 32 | 224 | 4 | 14 | 1098 |

Closest observed RTL single-token baseline from Stage 6E step0:

| Config | Q | K | V | QKV quant | Attention | Concat | W_O | Total | Projection PE stall | Attention PE stall | SFU stall | Output stall | Model delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H1/D8 | 186 | 186 | 186 | 48 | 88 | 17 | 197 | 884 | 0 | 22 | 6 | 0 | +696 |
| H2/D8 | 690 | 690 | 690 | 96 | 176 | 34 | 709 | 3021 | 1216 | 44 | 12 | 1 | +2593 |
| H4/D8 | 2658 | 2658 | 2658 | 192 | 352 | 68 | 2693 | 11135 | 7296 | 88 | 24 | 3 | +10035 |
| H2/D16 | 2658 | 2658 | 2658 | 192 | 330 | 68 | 2693 | 11111 | 7296 | 88 | 12 | 17 | +10013 |

The RTL rows are not strict no-stall measurements. They include real ready/valid
handshakes, DesignWare model latency, serialized projection PE stall cycles,
Stage 5 attention PE/SFU stall cycles, and the testbench output/done
backpressure schedule. A strict no-stall RTL counter run is unavailable in the
current accepted vector set.

Final cache-full cumulative Stage 6E counters:

| Config | Steps | Total | Q | K | V | QKV quant | Attention | Concat | W_O | Projection PE stall | Attention PE stall | SFU stall | Output stall | Peak seq |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H1/D8 | 8 | 9722 | 1674 | 1674 | 1674 | 432 | 2776 | 136 | 1577 | 0 | 736 | 300 | 4 | 8 |
| H2/D8 | 8 | 30417 | 6210 | 6210 | 6210 | 864 | 5552 | 272 | 5672 | 10640 | 1472 | 600 | 5 | 8 |
| H4/D8 | 8 | 105409 | 23922 | 23922 | 23922 | 1728 | 11104 | 544 | 21544 | 63840 | 2944 | 1200 | 11 | 8 |
| H2/D16 | 8 | 104105 | 23922 | 23922 | 23922 | 1728 | 9816 | 544 | 21544 | 63840 | 2944 | 600 | 34 | 8 |

## 9. External Interface Freeze

Final top: `rtl/attention/projection_integrated_mha.sv`.

Weight load:

- `weight_valid`, `weight_ready`
- `weight_kind`
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

Freeze rules:

- Later stages should consume Stage 6 through a wrapper or layer-level adapter.
- Do not directly change Stage 6 ready/valid semantics.
- Do not change weight layout `weight[kind][output_index][input_index]`.
- Do not change QKV/concat numeric boundaries or FP32-to-FP16 policy.
- Do not change current-token causal semantics.
- Do not change all-head atomic commit.
- Do not accept the next hidden token before final done.
- If a later stage must modify any frozen behavior, rerun the full Stage 5 and
  Stage 6 regression set and update specs/reports together.

## 10. Acceptance Decision

Stage 6 is accepted for projection-integrated MHA correctness with the
conditional audit notes above. The accepted tag remains on the final Stage 6
correctness commit, not on this audit report.

Do not infer that a complete Transformer layer is finished, and do not treat the
behavioral memories or cycle counters as physical implementation, timing, area,
power, WNS, STA, or layout results.
