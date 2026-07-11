# Stage 1 Summary

Status: CONDITIONAL PASS after Docker VCS/DC closure; FP MAC numeric route remains proposed, not frozen.

Implemented ready/valid stream primitives, FIFO, behavioral SRAM wrappers, signed-integer helper arithmetic wrappers, Python reference models, tests, VCS RTL simulation entry points, vlogan lint entry, and DC elaboration checks.

## Key Updates
- Added real VCS RTL simulation for all implemented Stage 1 RTL modules in `tb/rtl/stage1/tb_stage1_all.sv`.
- Added `make stage1-rtl-sim`.
- Updated Stage 1 simulation/lint/synthesis scripts to run Synopsys tools from `build/` and keep reports concise.
- Added DesignWare VCS and DC probes under `scripts/sim/run_stage1_dw_probe.sh` and `scripts/synth/stage1_dw_probe.tcl`.
- Added `reports/stage_01/docker_closure.md`.

## Verification Run
- Host `python -m pytest tb\model tb\unit`: 26 passed.
- Host `python scripts\sim\run_stage1_tests.py`: pytest passed, py_compile passed, RTL sim skipped because host has no VCS.
- Docker `make stage1-rtl-sim`: PASS; VCS compile/run exit 0; RTL assertions executed; assertion/TB error count 0.
- Docker `make stage1-lint`: PASS; static hygiene passed; vlogan compile exit 0; no vlogan diagnostics; Verilator not found.
- Docker `make stage1-synth`: PASS; DC analyze/elaborate/link/check_design exit 0.
- Docker `bash scripts/sim/run_stage1_dw_probe.sh`: PASS for probed DW FP16 multiply, FP32 add/MAC, EXP, division, reciprocal, SQRT, and reciprocal SQRT.
- Docker DW DC probe: PASS for non-conversion DW floating-point instances.

## Environment Limitations
- Docker `make stage0-test` failed because `pytest` is not installed in the container.
- Docker `make stage1-test` failed because the container has Python 3.6.9, which cannot parse repo files using `from __future__ import annotations`; its VCS RTL substep still passed.
- No PDK, standard-cell target library, SRAM macro, or memory compiler output was used.
- DC results are elaboration/synthesizability checks only and are not area, power, frequency, WNS, STA, or paper-comparable PPA.

## Floating-Point Status
- DesignWare FP16 multiply, FP32 add, FP32 MAC, EXP, division, reciprocal, SQRT, and reciprocal SQRT were found usable in minimal VCS and DC probes.
- DW FP16/FP32 conversion path compiled but did not match simple IEEE bit-conversion expectations in the probe; conversion semantics remain unconfirmed.
- Recommended `PROPOSED` FP MAC path: finite-only FP16-to-FP32 conversion boundary plus FP32 `DW_fp_mac` or FP32 multiply/add in a ready/valid wrapper.
- Final FP MAC interface, fused/non-fused behavior, exception/special-value policy, and latency are not frozen.

Stage 2 may reuse Stage 1 stream/FIFO/SRAM infrastructure and integer helper modules for scaffolding, but final PE/Softmax/Attention numeric RTL must not start until FP MAC semantics, interface, and latency are confirmed.
