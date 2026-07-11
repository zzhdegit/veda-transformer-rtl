# Stage Handoff

## Stage
Stage 1: Arithmetic and Memory Primitives

## Input
- `AGENTS.md`
- `PROJECT_STATE.md`
- `docs/stage_00_spec.md`
- `transformer_rtl_plan_md/01_arithmetic_and_memory_primitives.md`
- Stage 0 software reference model and tests
- `README_TOOL_USAGE.md` for local Docker usage
- User request for Stage 1 Docker toolchain closure and floating-point feasibility checks dated 2026-07-11

## Completed
- Implemented reusable ready/valid primitives:
  - `stream_reg`
  - `skid_buffer`
- Implemented memory primitives:
  - `sync_fifo`
  - `sram_1p_wrapper`
  - `sram_2p_wrapper`
- Implemented signed two's-complement integer/fixed-point helper arithmetic wrappers:
  - `mul_unit`
  - `add_unit`
  - `mac_unit`
  - `compare_max`
  - `round_sat`
- Added Python reference models for implemented arithmetic and memory behavior.
- Added pytest/static tests for Stage 0 and Stage 1 reference behavior and RTL hygiene.
- Added VCS SystemVerilog testbench `tb/rtl/stage1/tb_stage1_all.sv`.
- Added `make stage1-rtl-sim`.
- Added/updated scripts for Stage 1 VCS simulation, vlogan lint, DC elaboration, and DesignWare probes.
- Ran real Docker VCS simulation of Stage 1 RTL with assertions enabled.
- Ran Docker vlogan compile/lint and DC analyze/elaborate/check_design.
- Ran DesignWare VCS and DC probes for local floating-point IP feasibility.
- Added `reports/stage_01/docker_closure.md`.

## Files added or modified
- `PROJECT_STATE.md`
- `HANDOFF.md`
- `Makefile`
- `rtl/common/stream_reg.sv`
- `rtl/common/skid_buffer.sv`
- `rtl/memory/sync_fifo.sv`
- `rtl/memory/sram_1p_wrapper.sv`
- `rtl/memory/sram_2p_wrapper.sv`
- `rtl/arithmetic/mul_unit.sv`
- `rtl/arithmetic/add_unit.sv`
- `rtl/arithmetic/mac_unit.sv`
- `rtl/arithmetic/compare_max.sv`
- `rtl/arithmetic/round_sat.sv`
- `model/arithmetic/__init__.py`
- `model/arithmetic/numeric_format.py`
- `model/arithmetic/arithmetic_reference.py`
- `model/memory/__init__.py`
- `model/memory/fifo_reference.py`
- `tb/model/test_stage1_arithmetic.py`
- `tb/model/test_stage1_memory.py`
- `tb/unit/test_stage1_rtl_static.py`
- `tb/rtl/stage1/tb_stage1_all.sv`
- `scripts/sim/run_stage1_tests.py`
- `scripts/sim/run_stage1_vcs.py`
- `scripts/sim/run_stage1_vcs.sh`
- `scripts/sim/run_stage1_dw_probe.sh`
- `scripts/lint/run_stage1_lint.py`
- `scripts/synth/run_stage1_synth_check.py`
- `scripts/synth/stage1_elaborate.tcl`
- `scripts/synth/stage1_dw_probe.tcl`
- `scripts/env.example.sh`
- `reports/stage_01/test_results.txt`
- `reports/stage_01/lint_results.txt`
- `reports/stage_01/synth_check.txt`
- `reports/stage_01/vcs_rtl_sim.txt`
- `reports/stage_01/dw_probe.txt`
- `reports/stage_01/dw_probe_dc.log`
- `reports/stage_01/docker_closure.md`
- `reports/stage_01/summary.md`

## Frozen interfaces and decisions
- All implemented stream and arithmetic wrappers use `clk`, active-low `rst_n`, input ready/valid, output ready/valid, payload, metadata, and `last`.
- Ready/valid transfer occurs only on `valid && ready`.
- Payload, metadata, and `last` must remain stable while `valid=1` and `ready=0`.
- Reset clears valid state for all implemented stream, FIFO, SRAM response, and arithmetic output pipelines.
- `stream_reg` latency: 1 output register cycle after input acceptance when downstream is ready. Initiation interval: 1 without backpressure.
- `skid_buffer` latency: registered output behavior. Accepted input appears after a clock edge; after a buffered stall release, the buffer may spend one cycle moving the skid entry to the output before accepting a new item. It preserves order and uses local state for `in_ready` so it cuts long ready paths.
- `sync_fifo` latency: no empty fall-through; data written to an empty FIFO becomes readable on the next cycle. Same-cycle pop/push preserves order and allows full pop+push. Non-power-of-two depths are supported.
- `sync_fifo` outputs `full`, `empty`, `almost_full`, and `occupancy`. `almost_full` is `occupancy >= ALMOST_FULL_THRESHOLD`.
- `sram_1p_wrapper` read latency: explicit `READ_LATENCY=1`. It is synchronous read. Write requests do not emit a read response. Reset does not clear memory contents.
- `sram_2p_wrapper` read latency: explicit `READ_LATENCY=1`. It is 1R1W. Same-cycle same-address read/write is read-first, returning the old value. Reset does not clear memory contents.
- SRAM wrappers are behavioral macro-replacement boundaries only and must not be used for real SRAM area or timing claims.
- Arithmetic wrappers currently support signed two's-complement integer/fixed-point helper behavior only.
- `mul_unit`, `add_unit`, `mac_unit`, `compare_max`, and `round_sat` latency: 1 output register cycle after input acceptance when downstream is ready. Initiation interval: 1 without backpressure.
- `round_sat` supports arithmetic right shift truncation and nearest-even fixed-point rounding before signed saturation/truncation. Nearest-even uses sign-extended magnitude for negative inputs. It is not an FP16 converter.

## Verification performed
- Host `python -m pytest tb\model tb\unit`: 26 passed.
- Host `python scripts\sim\run_stage1_tests.py`: 26 tests passed, Python compile passed, RTL sim skipped because the host has no VCS.
- Host `python scripts\lint\run_stage1_lint.py`: static RTL hygiene passed.
- Docker `make stage0-test`: failed because container `pytest` is not installed.
- Docker `make stage1-test`: failed because container Python is 3.6.9 and cannot parse repo files using `from __future__ import annotations`; the VCS RTL substep still passed.
- Docker `make stage1-rtl-sim`: passed. VCS compile/run exit 0, pass marker present, assertion/TB error count 0.
- Docker `make stage1-lint`: passed. Static hygiene passed; vlogan compile exit 0; no vlogan diagnostics; Verilator not found.
- Docker `make stage1-synth`: passed. DC analyze/elaborate/link/check_design exit 0 for Stage 1 RTL tops.
- Docker `bash scripts/sim/run_stage1_dw_probe.sh`: passed for the DesignWare VCS probe.
- Docker DW DC probe: passed for non-conversion floating-point DesignWare instances.

## Commands to reproduce
Host:

```bash
python -m pytest tb\model tb\unit
python scripts\sim\run_stage1_tests.py
python scripts\lint\run_stage1_lint.py
```

Docker:

```bash
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage0-test'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage1-test'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage1-rtl-sim'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage1-lint'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage1-synth'
docker exec nailong bash -lc 'cd /workspace/VEDA && bash scripts/sim/run_stage1_dw_probe.sh'
docker exec nailong bash -lc 'cd /workspace/VEDA/build/stage1_dw_probe && dc_shell -f /workspace/VEDA/scripts/synth/stage1_dw_probe.tcl | tee /workspace/VEDA/reports/stage_01/dw_probe_dc.log'
```

## EDA environment and limitations
- Docker instructions used: `README_TOOL_USAGE.md`; container `nailong`; repository path `/workspace/VEDA`.
- `vcs`: found, VCS-MX `O-2018.09-SP2-2_Full64` in compile/run logs.
- `vlogan`: found, compile exit 0.
- `verdi`: found, not used for noninteractive closure.
- `dc_shell`: found, DC startup reported `L-2016.03-SP1`.
- `pt_shell`: found, `M-2016.12-SP1`; not used because no target library exists.
- `fc_shell`, `verilator`, `iverilog`, `gmake`, `pytest`, and container `python` were not found.
- Container `python3` is `3.6.9`, which is too old for the current Python sources.
- DesignWare simulation files and `dw_foundation.sldb` were visible in the installed Synopsys tree.
- No license error was observed in the executed VCS/DC probes.
- No PDK, standard-cell target library, SRAM macro, memory compiler output, layout database, P&R, DRC, LVS, or STA was used.
- DC results are only analyze/elaborate/check_design checks. They are not PPA and cannot support area, power, frequency, WNS, or paper-metric claims.

## DesignWare floating-point status
- `DW_fp_mult #(10, 5, 1)` for FP16 multiply: VCS PASS, DC PASS for finite normal probe cases.
- `DW_fp_add #(23, 8, 1)` for FP32 add: VCS PASS, DC PASS for finite normal probe cases.
- `DW_fp_mac #(23, 8, 1)` for FP32 MAC: VCS PASS, DC PASS for finite normal probe cases.
- `DW_fp_exp #(23, 8, 1, 2)`: VCS PASS, DC PASS for `exp(0.0)`.
- `DW_fp_div #(23, 8, 1, 0)`: VCS PASS, DC PASS for `4.0 / 2.0`.
- `DW_fp_recip #(23, 8, 1, 0)`: VCS PASS, DC PASS for `1 / 2.0`.
- `DW_fp_sqrt #(23, 8, 1)`: VCS PASS, DC PASS for `sqrt(4.0)`.
- `DW_fp_invsqrt #(23, 8, 1)`: VCS PASS, DC PASS for `1/sqrt(4.0)`.
- `DW_fp_ifp_conv` plus `DW_ifp_fp_conv` conversion paths compiled, but simple IEEE FP16/FP32 conversion expectations were not met in the probe. Do not treat the conversion path as confirmed.

## FP MAC candidate routes
- Option A: FP16 operands -> FP16 `DW_fp_mult` product -> product converted to FP32 -> FP32 accumulation. This uses confirmed FP16 multiply but rounds product at FP16 precision and still needs a verified conversion boundary.
- Option B: FP16 operands -> finite FP16-to-FP32 conversion -> FP32 multiply -> FP32 accumulation. This best matches the FP32-equivalent accumulation direction, but requires a verified converter and selected FP32 multiply/add or MAC wrapper.
- Option C: FP16 operands -> finite FP16-to-FP32 conversion -> FP32 `DW_fp_mac` wrapper. This is executable in local probes, but fused/non-fused semantics and wrapper latency must be frozen.

Recommended `PROPOSED` route: finite-only FP16-to-FP32 conversion boundary plus FP32 `DW_fp_mac` or FP32 multiply/add inside a ready/valid wrapper. The currently probed DW conversion path is not sufficient to freeze conversion semantics.

## Known limitations
- Final FP MAC interface, fused/non-fused behavior, latency, initiation interval, rounding, saturation/clamp behavior, and NaN/Inf/denormal policy are not frozen.
- Bit-accurate FP16/FP32 formal RTL wrappers are not implemented.
- NaN, Inf, denormal, signed-zero, FP exception flags, FP saturation/clamp, and FP underflow behavior remain unsupported in committed Stage 1 arithmetic RTL.
- The implemented arithmetic helpers support signed integer/fixed-point ranges only, as set by module parameters.
- `round_sat` fixed-point nearest-even behavior is for signed integer magnitudes and is not IEEE-754 rounding.
- No independent flush input is implemented on the Stage 1 primitives; reset clears in-flight valid state.
- SRAM wrapper memory contents are not reset; upper layers must manage validity.
- SRAM macro read latency and physical collision behavior remain unconfirmed for real macros.

## Open issues
- Confirm whether the FP MAC should use Option B or Option C.
- Freeze fused versus non-fused MAC semantics.
- Freeze FP MAC latency and ready/valid wrapper initiation interval.
- Resolve finite FP16-to-FP32 and FP32-to-FP16 conversion implementation.
- Confirm NaN/Inf/denormal/signed-zero behavior for first numeric RTL.
- Provide real target libraries before any synthesis/STA/PPA claim.
- Provide SRAM macro or memory compiler information before physical SRAM binding.

## Requirements for Stage 2
- Stage 2 can directly reuse `stream_reg`, `skid_buffer`, `sync_fifo`, `sram_1p_wrapper`, and `sram_2p_wrapper` for control/dataflow scaffolding.
- Stage 2 can reuse signed integer helper arithmetic only for integer/fixed-point auxiliary paths, not final FP attention math.
- Stage 2 must not start final PE, Softmax, or Attention numeric RTL until FP MAC interface, semantics, latency, and bit-accurate software models are selected and verified.
- Stage 2 must preserve ready/valid stability, metadata alignment, reset-valid clearing, FIFO ordering, and SRAM validity-management rules from this handoff.
- Stage 2 must continue to run RTL assertions in VCS or another real SystemVerilog simulator before claiming RTL verification complete.
