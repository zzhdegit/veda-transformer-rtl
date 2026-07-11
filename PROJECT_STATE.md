# Project State

## Current stage
- Stage: 1
- Status: CONDITIONAL PASS after Docker VCS/DC closure. Stage 1 infrastructure RTL is simulated with assertions and DC-elaborated; final FP MAC semantics/interface/latency remain proposed and require user confirmation.
- Base commit: N/A - no git repository detected
- Last update: 2026-07-11 Docker toolchain closure and DesignWare feasibility probe

## Frozen decisions
- Current project line does not implement voting-based KV cache eviction.
- First implementation target is high-quality single-head generation attention.
- Attention math is `raw_score_i = dot(q, K_i)`, `scaled_score_i = raw_score_i / sqrt(d)`, `p_i = exp(scaled_score_i - max_score) / exp_sum`, and `o = sum_i p_i * V_i`.
- `qK^T` uses inner-product dataflow.
- Softmax uses element-serial reduction followed by element-serial normalization.
- `s'V` uses outer-product dataflow.
- K/V cache layout is token-major: `K_cache[token][dimension]` and `V_cache[token][dimension]`.
- No physical transposed K cache is allowed for the first version.
- Module streams use ready/valid.
- Ready/valid rule: payload, metadata, and `last` remain stable while `valid=1` and `ready=0`.
- Reset rule for implemented Stage 1 stream/pipeline modules: reset clears all in-flight valid state; memory arrays are not cleared.
- Stage 1 `stream_reg`: one registered output stage, latency 1 accepted transaction when downstream is ready, initiation interval 1 without backpressure.
- Stage 1 `skid_buffer`: two-entry elastic buffer with locally registered ready behavior; it cuts combinational ready propagation and preserves order. Output is registered, so accepted input appears after a clock edge; after a buffered stall release it may spend one cycle moving the skid entry to the output before accepting a new item.
- Stage 1 `sync_fifo`: parameterized `DATA_W`, `META_W`, and arbitrary positive `DEPTH`; non-power-of-two depths are supported; no fall-through from empty; same-cycle pop/push is supported, including full pop+push without overflow.
- Stage 1 SRAM wrappers are behavioral replacement boundaries only. They do not represent real SRAM macro PPA.
- Stage 1 `sram_1p_wrapper`: explicit `READ_LATENCY=1`, synchronous read, one request port, write requests produce no read response, clock enable gates requests/responses, reset does not clear memory.
- Stage 1 `sram_2p_wrapper`: explicit `READ_LATENCY=1`, 1-read/1-write behavioral memory, read-first same-address read/write behavior, clock enable gates requests/responses, reset does not clear memory.
- Stage 1 committed arithmetic RTL supports only signed two's-complement integer/fixed-point helper behavior. Implemented wrappers have one registered output stage and preserve metadata/`last`.
- Stage 1 did not add PDK, standard-cell libraries, SRAM macro files, DesignWare library files, or any licensed EDA library content to the repository.

## Provisional decisions
- First-version input and weight format target remains FP16.
- MAC accumulation target remains FP32 or an equivalent higher-precision synthesizable format.
- Softmax internal target remains FP32-equivalent for scaled scores, max, exponent sum, probabilities, and the output accumulation feeding FP16 output conversion.
- Rounding to FP16 output should use round-to-nearest-even unless the selected arithmetic IP forces a different documented behavior.
- Full IEEE NaN/Inf/denormal exception handling remains out of scope for the first version, and finite inputs are assumed until a later numeric spec expands this.
- DesignWare feasibility probes confirmed usable local instances for FP16 multiply, FP32 add, FP32 MAC, EXP, division, reciprocal, SQRT, and reciprocal SQRT, but those IPs are not yet integrated into formal RTL wrappers.
- The local DesignWare FP16/FP32 conversion probe compiled but did not match simple IEEE bit-conversion expectations; conversion semantics remain unconfirmed.
- Proposed FP MAC route: finite-only FP16-to-FP32 conversion boundary followed by FP32 `DW_fp_mac` or FP32 multiply/add accumulation inside ready/valid wrappers.
- Final FP MAC interface, fused versus non-fused behavior, numerical semantics, special-value policy, and latency are not frozen.
- SRAM macro availability, width, depth, ports, and real macro latency remain unknown until external PDK/memory inputs are available.

## Completed
- Stage 0 and Stage 1 Python/static regression passed on the host: `python -m pytest tb\model tb\unit` reported 26 passed.
- Implemented `rtl/common/stream_reg.sv`.
- Implemented `rtl/common/skid_buffer.sv`.
- Implemented `rtl/memory/sync_fifo.sv`.
- Implemented `rtl/memory/sram_1p_wrapper.sv`.
- Implemented `rtl/memory/sram_2p_wrapper.sv`.
- Implemented signed integer helper wrappers: `mul_unit`, `add_unit`, `mac_unit`, `compare_max`, and `round_sat`.
- Independent Stage 1 audit fixed `round_sat` nearest-even negative-input magnitude handling to sign-extend before taking the two's-complement absolute value.
- Added local assertions for valid/data/metadata stability, FIFO overflow/underflow prevention, FIFO occupancy range, transaction conservation, unknown output checks, and parameter legality.
- Added Stage 1 Python reference models for arithmetic and memory behavior.
- Added Stage 1 pytest coverage for arithmetic edge cases, FIFO ordering/backpressure, SRAM read latency, and read-first collision behavior.
- Added static RTL hygiene tests for required modules, forbidden `real`/`shortreal`, assertion coverage tokens, and PDK/macro artifact exclusion.
- Added VCS SystemVerilog testbench `tb/rtl/stage1/tb_stage1_all.sv`.
- Added `make stage1-rtl-sim`.
- Docker VCS RTL simulation passed for all implemented Stage 1 modules, with RTL assertions executed and no assertion/TB errors.
- Docker vlogan strict RTL compile passed with no diagnostics.
- Docker DC analyze/elaborate/link/check_design passed for all implemented Stage 1 RTL tops.
- Docker DesignWare VCS and DC probes passed for FP16 multiply, FP32 add/MAC, EXP, division, reciprocal, SQRT, and reciprocal SQRT.
- Added reports under `reports/stage_01/`, including `docker_closure.md`.

## Open issues
- This directory is not a git repository, so branch, commit, and working-tree cleanliness are not available.
- Docker Python regression is not closed: the container lacks `pytest`, and its Python 3.6.9 cannot parse repo files using `from __future__ import annotations`.
- DesignWare FP conversion parameterization and bit-exact semantics are not resolved; do not use the probed conversion path as a frozen FP16/FP32 converter.
- Final FP MAC interface, fused/non-fused behavior, numerical semantics, special-value policy, and latency need user confirmation.
- No real target `.lib` or SRAM macro information is available; no valid synthesis, STA, area, frequency, or power conclusion exists.
- Stage 2 may reuse stream/FIFO/SRAM wrappers and integer helpers for scaffolding or auxiliary fixed-point paths, but must not start final PE/Softmax/Attention numeric RTL until FP MAC decisions are frozen and modeled.

## Next action
- User should confirm the FP MAC route: recommended `PROPOSED` route is finite-only FP16-to-FP32 conversion plus FP32 `DW_fp_mac` or FP32 multiply/add accumulation.
- Freeze FP MAC wrapper latency, initiation interval, metadata behavior, fused/non-fused arithmetic semantics, rounding mode, saturation/clamp behavior, and NaN/Inf/denormal policy.
- Implement and verify a bit-exact finite FP16-to-FP32 converter or resolve the correct local DesignWare conversion configuration.
- Provide SRAM macro or memory compiler information before treating SRAM wrappers as physical memories.
