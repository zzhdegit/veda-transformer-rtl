# Project State

## Current stage
- Stage: 3
- Status: STAGE 3 PASS
- Branch observed: `stage2-pe-core`
- Requested Stage 3 branch: `stage3-single-head-attention`
- Last update: 2026-07-12

## Preflight deviations
- The requested branch `stage3-single-head-attention` was not present at Stage 3 start.
- The working tree was not clean at Stage 3 start because Stage 1B and Stage 2 deliverables were already uncommitted.
- The requested tag `stage2-correctness-accepted` was not present.
- Stage 3 proceeded on the existing verified Stage 2 working tree to avoid discarding or rewriting uncommitted baseline files.

## Frozen decisions
- Current project line does not implement voting-based KV cache eviction.
- First implementation target is high-quality single-head generation attention.
- Attention math is `raw_score_i = dot(q, K_i)`, `scaled_score_i = raw_score_i / sqrt(d)`, `p_i = softmax(scaled_score_i)`, and `o = sum_i p_i * V_i`.
- `qK^T` uses inner-product dataflow through `reconfigurable_pe_core`.
- `s'V` uses outer-product dataflow through `reconfigurable_pe_core`.
- K/V cache layout remains token-major: `K_cache[token][dimension]` and `V_cache[token][dimension]`.
- No physical transposed K cache is introduced.
- Module streams use ready/valid. A transfer occurs only on `valid && ready`.
- Payload, metadata, and `last` remain stable while `valid=1 && ready=0`.
- Reset clears in-flight valid state for implemented stream, pipeline, buffer response, SRAM response, arithmetic wrapper, PE core output logic, and attention controller output/done logic.
- Stage 1 infrastructure modules remain frozen as documented in `HANDOFF.md`: `stream_reg`, `skid_buffer`, `sync_fifo`, `sram_1p_wrapper`, `sram_2p_wrapper`, and signed integer helper arithmetic wrappers.
- Stage 1 SRAM wrappers are behavioral macro-replacement boundaries only. They do not represent real SRAM macro PPA.
- Stage 1B FP16 input layout is IEEE-like: 1 sign bit, 5 exponent bits, and 10 fraction bits.
- Stage 1B FP16-to-FP32 policy remains: normal finite FP16 values exactly expand to FP32, signed zero is preserved, FP16 subnormals flush to signed zero with `underflow_or_ftz=1`, and FP16 NaN/Inf inputs produce defined signed-zero output with `invalid=1` and trigger the non-synthesis assertion.
- Stage 1B/2/3 FP32 DesignWare use is confined to project-owned wrappers.
- `fp32_mac_wrapper` uses local `DW_fp_mac #(23, 8, 1)`, fused semantics, and local RNE code `rnd=3'b100`.
- `fp32_add_wrapper` uses local `DW_fp_add #(23, 8, 1)` and local RNE code `rnd=3'b100`.
- `fp32_exp_wrapper` uses local `DW_fp_exp #(23, 8, 1, 2)` behind a ready/valid wrapper and clamps finite inputs below `-20.0` (`32'hC1A00000`) to `+0.0`.
- `fp32_recip_wrapper` computes reciprocal through local `DW_fp_div #(23, 8, 1, 0)` as `1.0 / x` behind a ready/valid wrapper.
- Stage 2 balanced reduction order remains fixed as `(0+1),(2+3),...` at each level, then the next level in ascending pair order.
- Stage 2 dimension tiling order remains tile arrival order. Inner-product tile accumulation is sequential: `(((0 + tile0) + tile1) + ...)`.
- Stage 2 PE core remains correctness-first and transaction-serial. Stage 3 does not assume top-level PE II=1.
- No PDK, standard-cell libraries, SRAM macro files, DesignWare library files, or licensed EDA library content are committed to the repository.
- No formal area, power, frequency, WNS, STA, P&R, or PPA conclusion exists for Stage 1, Stage 1B, Stage 2, or Stage 3.

## Stage 3 completed
- Implemented arithmetic wrappers:
  - `rtl/arithmetic/fp32_exp_wrapper.sv`
  - `rtl/arithmetic/fp32_recip_wrapper.sv`
- Implemented Stage 3 attention RTL:
  - `rtl/attention/attention_score_scaler.sv`
  - `rtl/attention/score_buffer.sv`
  - `rtl/attention/softmax_reduction.sv`
  - `rtl/attention/softmax_normalization.sv`
  - `rtl/attention/single_head_attention_controller.sv`
  - `rtl/attention/single_head_attention.sv`
- Implemented Stage 3 Python models:
  - `model/attention/softmax_reference.py`
  - `model/attention/single_head_reference.py`
  - `model/attention/single_head_cycle_model.py`
- Implemented Stage 3 tests, vector generator, VCS runner, lint runner, DC elaboration script, and Makefile targets:
  - `tb/model/test_stage3_attention.py`
  - `tb/rtl/stage3/tb_fp32_exp_recip_wrappers.sv`
  - `tb/rtl/stage3/tb_single_head_attention.sv`
  - `scripts/sim/gen_stage3_vectors.py`
  - `scripts/sim/run_stage3_tests.py`
  - `scripts/sim/run_stage3_vcs.sh`
  - `scripts/lint/run_stage3_lint.py`
  - `scripts/synth/stage3_elaborate.tcl`
  - `scripts/synth/run_stage3_synth_check.py`
  - `make stage3-test`
  - `make stage3-rtl-sim`
  - `make stage3-lint`
  - `make stage3-synth`
- Added reports under `reports/stage_03/`.

## Stage 3 architecture
- Top-level first-version interface loads q/K/V into internal behavioral memories, then starts one single-head attention operation.
- `single_head_attention` outputs FP32 head vector tiles with `output_base_dim`, `output_lane_mask`, metadata, status, invalid, and `output_last`.
- `done_valid` reports operation completion after all output tiles are accepted.
- `D_HEAD` is an elaboration parameter. Verified RTL elaborations are `D_HEAD=8` and `D_HEAD=16` with `PE_NUM=8`.
- Python bit-model tests cover `d_head = 1, 7, 8, 9, 13, 16` and `seq_len = 1, 2, 3, 7, 8, 15, 31, 32`.
- Score buffer stores scaled FP32 scores in token order. `READ_LATENCY=1`; memory contents are not reset, and valid length/read count prevent uninitialized reads.
- Probability buffer stores normalized FP32 probabilities so `D_HEAD > PE_NUM` can reuse probabilities for multiple SV output-dimension tiles without changing the Stage 2 PE core interface.
- Online softmax reduction is correctness-first and serial:
  - first score initializes `m = score`, `z = 1.0`;
  - later scores use `m_new = max(m_old, x)`;
  - `z_new = z_old * exp(m_old - m_new) + exp(x - m_new)`.
- Normalization computes one reciprocal of `exp_sum`, then serially computes `p_i = exp(score_i - max_final) * inv_sum`.
- SV scheduling processes one output dimension tile at a time. For each tile, it reads all probabilities in token order and sends `p_i` with aligned `V_i` to `reconfigurable_pe_core` in `MODE_SV_OUTER`.

## Verification performed
- Host `python scripts/sim/run_stage3_tests.py`: PASS.
  - Stage 1B/2/3 model tests: 17 passed.
  - Python compile: PASS.
  - Host VCS skipped because host has no `vcs`.
- Docker `make stage3-rtl-sim`: PASS.
  - `fp32_exp_recip`: PASS.
  - `single_head_attention_d8`: PASS.
  - `single_head_attention_d16`: PASS.
  - VCS assertions were enabled with `-assert svaext`.
- Docker `make stage3-lint`: PASS; static hygiene passed and vlogan diagnostics were none.
- Docker `make stage3-synth`: PASS; DC analyze/elaborate/link/check_design passed for default `D_HEAD=8`, parameterized `D_HEAD=16`, and `score_buffer DEPTH=4096`.
- Stage 2 regression after Stage 3 remained passing:
  - Host `python scripts/sim/run_stage2_tests.py`: PASS.
  - Docker `make stage2-rtl-sim`: PASS.
  - Docker `make stage2-lint`: PASS.
  - Docker `make stage2-synth`: PASS.

## Stage 3 latency results
RTL VCS reported:

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

These are functional RTL cycle statistics only, not PPA.

## Known limitations
- Stage 3 does not implement dynamic KV append, KV Cache Manager, multi-head attention, QKV projection, output projection, full Transformer, voting, P&R, or formal PPA.
- Stage 3 uses internal behavioral memories for q/K/V load and verification. They are not SRAM macro PPA.
- Probability buffering is used to support `D_HEAD > PE_NUM` with a single unchanged Stage 2 PE core; this is correctness-first and not the final throughput schedule.
- Softmax reduction and normalization are serial and do not hide SFU latency.
- The current FP32 MAC/add/EXP/div wrappers contain combinational DesignWare arithmetic before output registers; real timing closure likely requires pipelining once a target library and timing constraints exist.
- Full `single_head_attention` DC check was not run with `MAX_SEQ_LEN=4096` because that would elaborate large behavioral K/V arrays as non-macro storage. `score_buffer DEPTH=4096` was elaborated.

## Next action
- Stage 4 may add dynamic KV append/cache-manager interfaces around the Stage 3 single-operation attention engine.
- Stage 4 must preserve token-major K/V layout, score/probability token order, ready/valid contracts, FP16 FTZ policy, FP32 wrapper boundaries, and Stage 2 PE transaction-serial behavior unless a documented interface change is explicitly approved.
- Do not start Stage 4 until requested.
