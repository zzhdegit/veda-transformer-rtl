# Stage 7 Acceptance Audit

## 1. Audit Result

- Result: PASS
- Date: 2026-07-13
- Branch: `stage7-prenorm-transformer-layer`
- Final functional commit: `8ad948343179d2ae801fcea3679dce6b143382c3`
- Acceptance tag: `stage7-correctness-accepted`, target `8ad948343179d2ae801fcea3679dce6b143382c3`
- Remote sync: local branch matched `origin/stage7-prenorm-transformer-layer` before audit changes; audit commit and tag are to be pushed after this report.
- Working tree: source RTL/model/script files were clean before audit changes. Generated report files had pre-existing dirt and were refreshed by the requested regressions.

STAGE 7 ACCEPTANCE AUDIT PASS.

## 2. Accepted Functional Boundary

Stage 7 accepts one decoder-style Pre-Norm Transformer layer:

```text
input hidden FP16
-> exact FP16-to-FP32 expansion
-> RMSNorm1
-> normalized FP16
-> frozen Stage 6 projection-integrated MHA
-> MHA FP32 output
-> FP32 Residual1
-> RMSNorm2
-> normalized FP16
-> FFN W1
-> ReLU
-> FP32-to-FP16 activation quantization
-> FFN W2
-> FP32 Residual2
-> final tiled FP32 layer output
-> layer done
```

Frozen relation:

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

Stage 7 does not include multiple layers, LayerNorm, Post-Norm, GELU, SiLU,
SwiGLU, bias, dropout, RoPE, embedding, LM head, tokenizer, SRAM macro binding,
STA, layout, or formal PPA conclusions.

## 3. Git And Tag Audit

| Item | Result |
|---|---|
| Repository root | `D:/IC_Workspace/VEDA` |
| Current branch | `stage7-prenorm-transformer-layer` |
| Pre-audit branch tracking | `8ad9483` matched `origin/stage7-prenorm-transformer-layer` |
| Remote URL | `https://github.com/zzhdegit/veda-transformer-rtl.git`; no credential in URL |
| Final functional commit | `8ad948343179d2ae801fcea3679dce6b143382c3` |
| Stage 5 tag target | `stage5-correctness-accepted^{}` = `c350c47c067e224285956d8a1bcd4a469abda672`; not moved |
| Stage 6 tag target | `stage6-correctness-accepted^{}` = `c59ac45cd7bd59fbddaf486ea655b1d753342be9`; not moved |
| Stage 7 tag at audit start | absent locally and remotely |
| Stage 7 tag decision | create only after PASS, pointing to the final functional commit, not the audit-document commit |

No rebase, force push, hard reset, clean, old tag move, or history rewrite was
performed.

## 4. Restricted File Audit

Commands audited tracked files with `git ls-files` and a restricted-name scan.
The only tracked path matching the restricted-name expression was:

| Path | Category | Decision |
|---|---|---|
| `tb/rtl/stage1b/tb_dw_fp_mac_semantics.sv` | `DW_fp_` filename token | Testbench file name only; no Synopsys DesignWare source is tracked |

No tracked token, credential, password, secret, license, PDK, technology library,
EDA install directory, `.sldb`, `.db`, `.lib`, `.lef`, `.gds`, `.ndm`, `.nlib`,
build directory, `simv`, `csrc`, waveform, or large generated simulator artifact
was found.

Local-only files remain ignored:

- `LOCAL_GITHUB.md`: ignored by `.git/info/exclude`
- `transformer_rtl_plan_md/LATE_STAGE_REAL_MODEL_VALIDATION_PLAN.md`: ignored by `.git/info/exclude`

## 5. Regression Results

Host commands:

| Command | Result | Notes |
|---|---|---|
| `python scripts/sim/run_stage7a_tests.py` | PASS | 8 pytest tests, Python compile sweep, Stage 7 cycle examples |
| `python scripts/sim/run_stage7_tests.py` | unavailable | Unified Stage 7 Python command does not exist |
| `python scripts/sim/run_stage6_tests.py` | PASS | Stage 6A-D 16 tests each, Stage 6E 18 tests, vectors, compile sweep, cycle model |
| `python scripts/sim/run_stage5_tests.py` | PASS | 31 model tests; host VCS skipped because host `vcs` was unavailable |

Docker commands:

| Command | Result | Notes |
|---|---|---|
| `make stage7a-test` | PASS | Model and compile sweep |
| `make stage7b-test && make stage7b-rtl-sim && make stage7b-lint && make stage7b-synth` | PASS | D8/D16 RMSNorm/residual VCS, vlogan, DC structural checks |
| `make stage7c-test && make stage7c-rtl-sim && make stage7c-lint && make stage7c-synth` | PASS | D8/D16 FFN/ReLU VCS, vlogan, DC structural checks |
| `make stage7d-test && make stage7d-rtl-sim && make stage7d-lint && make stage7d-synth` | PASS | H1/D8, H2/D8, H4/D8, H2/D16 final-top VCS, reset audit, vlogan, DC structural checks |
| `make stage7-test && make stage7-rtl-sim && make stage7-lint && make stage7-synth` | unavailable | Unified Stage 7 make targets do not exist |
| `make stage6-test && make stage6-rtl-sim && make stage6-lint && make stage6-synth` | PASS | Stage 6 model, RTL, lint, DC compatibility |
| `make stage5-test && make stage5-rtl-sim && make stage5-lint && make stage5-synth` | PASS | Stage 5 model, RTL, lint, DC compatibility |

VCS was run with `-assert svaext` for Stage 7D RTL simulation. Stage 7B and
Stage 7D lint reported only DesignWare pragma-no-effect warnings. Stage 7C lint
reported no diagnostics. DC checks were analyze/elaborate/link/check_design only
and do not report area, power, frequency, WNS, STA, layout, or PPA.

## 6. Final Top Instance Audit

Final top: `rtl/transformer/transformer_layer.sv`.

This table counts the final top hierarchy. It excludes old-stage modules that
remain in the repository only for regression compatibility.

| Module | Final-top count | Notes |
|---|---:|---|
| `projection_integrated_mha` | 1 | The only Stage 6 MHA child |
| `rmsnorm_engine` | 2 | RMSNorm1 and RMSNorm2 |
| `residual_add_engine` | 2 | Residual1 and Residual2 |
| `ffn_engine` | 1 | W1 and W2 share this engine |
| `reconfigurable_pe_core` | 3 | Stage 6 projection, Stage 5 attention, Stage 7 FFN |
| `shared_gemv_projection_core` | 1 | Stage 6 projection GEMV |
| `fp32_mac_wrapper` | 29 | 24 PE-lane MACs, 2 RMSNorm MACs, 3 Stage 5 attention MACs |
| `fp32_add_wrapper` | 12 | PE reductions/accumulators, RMSNorm add-eps, residuals, softmax adders |
| `fp32_sqrt_wrapper` | 2 | RMSNorm1 and RMSNorm2 |
| `fp32_recip_wrapper` | 3 | RMSNorm1, RMSNorm2, softmax normalization |
| `fp32_exp_wrapper` | 2 | Stage 5 softmax reduction/normalization |
| `fp32_to_fp16` | 5 | RMSNorm1, RMSNorm2, FFN activation, Stage 6 QKV quantizer, Stage 6 concat quantizer |

Confirmed:

- exactly one Stage 6 MHA instance;
- RMSNorm1 and RMSNorm2 are distinct instances;
- Residual1 and Residual2 are distinct instances;
- one FFN instance shares one PE core for W1 and W2;
- no old test module enters the final top hierarchy;
- no extra full Stage 6 copy is instantiated.

## 7. Numeric Coverage Audit

| Config | Full-layer RTL | Dense MHA | Dense FFN | Nonuniform Gamma | Multi-token | Cache-full | Backpressure | Mid-phase reset |
|---|---|---|---|---|---|---|---|---|
| H1/D8 | Yes | No, identity/directed | No, directed/sparse | No, gamma all ones | No | No final-top RTL | Yes, final/top active-load checks | Yes, 14 final-top scenarios |
| H2/D8 | Yes | Python dense model; RTL identity/directed | Python dense model; RTL directed/sparse | Python dense model; RTL gamma all ones | Yes, RTL two-token and Python three-token | Python Stage 7 wrapper; not final-top RTL | Yes, final/top active-load checks | No final-top reset scenario |
| H4/D8 | Yes | No, identity/directed | No, directed/sparse | No, gamma all ones | No | No final-top RTL | Yes, final/top active-load checks | No final-top reset scenario |
| H2/D16 | Yes | No, identity/directed | No, directed/sparse | No, gamma all ones | No | No final-top RTL | Yes, final/top active-load checks | No final-top reset scenario |

Dense complete-layer coverage exists in the Stage 7 Python bit model for H2/D8:
dense WQ/WK/WV/WO, dense W1/W2, nonuniform gamma1, nonuniform gamma2, mixed-sign
hidden input, and a three-token sequence. Stage 7D final-top RTL vectors use
identity/directed Stage 6 MHA weights, directed/sparse FFN weights, gamma all
ones, and mixed-sign hidden inputs.

Intermediate Python trace nodes compared by model tests include
`input_fp32`, `norm1_sum_sq`, `norm1_inv_rms`, `norm1_fp32`, `norm1_fp16`,
`mha_fp32`, `residual1_fp32`, `norm2_sum_sq`, `norm2_inv_rms`, `norm2_fp32`,
`norm2_fp16`, `ffn1_fp32`, `relu_fp32`, `activation_fp16`, `ffn2_fp32`, and
`final_fp32`. Final-top RTL checks compare final tiled FP32 output, status,
metadata, done, and sequence length. High-precision model code is used only for
diagnostic error statistics, not as an RTL tolerance.

## 8. Reset And Backpressure Coverage

Stage 7D final-top directed reset audit was added to the testbench and run for
H1/D8. Every reset scenario interrupts an active transaction, checks
reset-visible outputs, reloads required state, and executes a clean recovery
token. The recovery path verifies no duplicate or missing commit via
`perf_generation_steps == 1`.

| Reset scenario | Result |
|---|---|
| reset during input load | PASS |
| reset during RMSNorm1 reduction | PASS |
| reset during RMSNorm1 apply | PASS |
| reset during MHA | PASS |
| reset during Residual1 | PASS |
| reset during RMSNorm2 reduction | PASS |
| reset during RMSNorm2 apply | PASS |
| reset during FFN1 | PASS |
| reset during ReLU | PASS |
| reset during activation quantization | PASS |
| reset during FFN2 | PASS |
| reset during Residual2 | PASS |
| reset during final output stall | PASS |
| reset during layer done stall | PASS |

The reset checks cover active transaction clearing, `output_valid` clearing,
`done_valid` clearing, metadata no-leak, no duplicate output, no duplicate
commit, clean token execution after reset and weight reload, Stage 6-style
`valid_seq_len` reset semantics, and no X on reset-visible valid/status outputs.

Backpressure and transaction coverage:

| Item | Coverage |
|---|---|
| Input backpressure | Stage 7D final top rejects `token_valid` during an active layer transaction |
| Weight-load backpressure | Stage 7D final top rejects `weight_valid` during an active layer transaction |
| MHA output backpressure | Covered in Stage 6 component/final MHA regressions; final Stage 7 top stores MHA output internally |
| FFN output backpressure | Covered in Stage 7C FFN regression; final Stage 7 top stores FFN output internally |
| Final output backpressure | Stage 7D final top stalls `output_ready` and checks stable payload until accepted |
| Final done backpressure | Stage 7D final top stalls `done_ready` and checks stable done payload |
| Two-token continuous run | Stage 7D H2/D8 two-token test checks sequence length 1 then 2 |
| Cache-full extra token | Stage 7 Python wrapper and Stage 6 RTL cover cache-full; Stage 7D final-top RTL cache-full is not separately run |
| Metadata propagation | Stage 7D final top checks output and done metadata |
| Next token only after layer done | Stage 7D two-token sequence and active-input rejection cover this externally |
| Stage 6 done vs Stage 7 layer done | Audited; Stage 6 done is internal, Stage 7 `done_valid` occurs only after residual2 final output |

## 9. Numeric Contract Audit

RMSNorm:

- sequential dimension-order fused MAC: PASS;
- exact power-of-two mean scaling: PASS;
- `EPS_FP32 = 32'h3727_C5AC`: PASS;
- sqrt then reciprocal: PASS;
- apply order `(x * inv_rms) * gamma`: PASS;
- output FP16: PASS;
- NaN/Inf invalid policy consistent between model and RTL: PASS.

Residual:

- FP32 add: PASS;
- Residual1 is not quantized: PASS;
- Residual2 outputs FP32: PASS;
- signed-zero behavior documented as exact zero to `+0`: PASS.

FFN:

- `D_FFN = 4 * D_MODEL`: PASS;
- W1/W2 output-row-major: PASS;
- no bias: PASS;
- W1 and W2 share one FFN PE core: PASS;
- ReLU negative finite and signed zero output `+0`: PASS;
- NaN/Inf invalid and output `+0`: PASS;
- ReLU output quantized to FP16 before W2: PASS;
- W2 output FP32: PASS.

Layer:

- Pre-Norm order is correct: PASS;
- Stage 6 interface and weight layout are preserved: PASS;
- current-token causal semantics remain owned by Stage 6: PASS;
- all-head atomic commit remains owned by Stage 6: PASS;
- post-commit Stage 7 invalid does not roll back K/V: PASS by contract;
- layer done is emitted only after final output completes: PASS.

## 10. Cycle Baseline

No-stall cycle model output from
`python model/transformer/transformer_layer_cycle_model.py`:

| Config | Input | Norm1 reduce | Norm1 apply | MHA | Residual1 | Norm2 reduce | Norm2 apply | FFN1 | ReLU/Quant | FFN2 | Residual2 | Output | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H1/D8 | 8 | 8 | 8 | 40 | 8 | 8 | 8 | 32 | 64 | 32 | 8 | 1 | 249 |
| H2/D8 | 16 | 16 | 16 | 144 | 16 | 16 | 16 | 128 | 128 | 128 | 16 | 2 | 666 |
| H4/D8 | 32 | 32 | 32 | 544 | 32 | 32 | 32 | 512 | 256 | 512 | 32 | 4 | 2076 |
| H2/D16 | 32 | 32 | 32 | 544 | 32 | 32 | 32 | 512 | 256 | 512 | 32 | 4 | 2076 |

`ReLU/Quant` is `perf_relu_cycles + perf_activation_quantization_cycles` in the
model table.

RTL counter availability:

| Config | Total | Norm stall | MHA stall | FFN PE stall | Arithmetic stall | Buffer stall | Output stall | Model delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H1/D8 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| H2/D8 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| H4/D8 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| H2/D16 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |

The Stage 7 top exposes counter ports, but the current Stage 7D VCS summaries do
not print per-run counter values. VCS simulation time is not an architectural
cycle measurement and is not used as a performance conclusion. Missing
observations: total layer cycle counter value per token, stall bucket counter
values per token, and per-phase RTL deltas.

## 11. External Interface Freeze

Clock/reset:

- `clk`
- `rst_n`

Weight load:

- `weight_valid`
- `weight_ready`
- `weight_kind`
- `weight_output_index`
- `weight_input_index`
- `weight_data_fp16`
- `weight_last`
- `weight_commit`

Hidden input:

- `token_valid`
- `token_ready`
- `token_dim`
- `token_hidden_fp16`
- `token_last_dim`
- `token_meta`

Final output:

- `output_valid`
- `output_ready`
- `output_base_dim`
- `output_vector_fp32`
- `output_lane_mask`
- `output_status`
- `output_invalid`
- `output_meta`
- `output_last`

Done/state:

- `done_valid`
- `done_ready`
- `done_status`
- `done_invalid`
- `done_meta`
- `done_valid_seq_len`
- `current_valid_seq_len`

Stage 7 counters:

- `perf_generation_steps`
- `perf_total_layer_cycles`
- `perf_input_load_cycles`
- `perf_norm1_reduce_cycles`
- `perf_norm1_apply_cycles`
- `perf_mha_cycles`
- `perf_residual1_cycles`
- `perf_norm2_reduce_cycles`
- `perf_norm2_apply_cycles`
- `perf_ffn1_cycles`
- `perf_relu_cycles`
- `perf_activation_quantization_cycles`
- `perf_ffn2_cycles`
- `perf_residual2_cycles`
- `perf_final_output_cycles`
- `perf_norm_stall_cycles`
- `perf_mha_stall_cycles`
- `perf_ffn_pe_stall_cycles`
- `perf_weight_stall_cycles`
- `perf_buffer_stall_cycles`
- `perf_output_stall_cycles`
- `perf_peak_valid_seq_len`

Freeze rules:

- Future optimization must use internal implementation changes or wrappers.
- Do not change Stage 7 external ready/valid semantics.
- Do not change `weight_kind` values or weight layout.
- Do not change Pre-Norm order.
- Do not change FP16/FP32 numeric boundaries.
- Do not change RMSNorm operation order.
- Do not change Stage 6 commit semantics.
- Do not change Layer done semantics.
- If any frozen rule changes, rerun full Stage 5/6/7 regressions and update the
  bit model and spec.

## 12. Known Limitations

- Stage 7 scheduling is serial correctness scheduling, not optimized throughput.
- Projection, concat, FFN, and cache memories remain behavioral memories; no
  SRAM macro or physical memory claim is made.
- Timing pipeline is provisional.
- Dense complete-layer coverage is Python bit-model coverage for H2/D8; Stage
  7D final-top RTL dense MHA plus dense FFN plus nonuniform gamma is not yet a
  separate RTL vector.
- RTL final-top internal-node observability is limited; final-top RTL compares
  final outputs/status/metadata/done, while model tests compare intermediate
  trace nodes.
- DC checks are structural analyze/elaborate/link/check_design only.
- No formal area, power, frequency, WNS, STA, P&R, layout, or PPA conclusion is
  produced.
- Stage 7 does not implement multiple layers, embedding, LM head, tokenizer, or
  complete language-model functionality.
- Unified Stage 7 make targets (`stage7-test`, `stage7-rtl-sim`,
  `stage7-lint`, `stage7-synth`) are not present; phase-level targets were used.

## 13. Acceptance Decision

All functional, regression, hierarchy, numeric-contract, reset, backpressure,
Git, tag, and restricted-file checks required for the Stage 7 correctness
baseline passed or are documented as non-acceptance limitations.

STAGE 7 ACCEPTANCE AUDIT PASS.
