# Stage 1 Docker Toolchain Closure

Date: 2026-07-11

## Local Docker Instructions Found
- `README_TOOL_USAGE.md` describes the local Docker flow.
- The documented Windows workspace root is mounted at `/workspace` in the container.
- The documented container is `nailong`.
- The command flow used was:
  - `docker ps -a`
  - `docker start nailong` if needed
  - `docker exec nailong bash -lc 'cd /workspace/VEDA && <command>'`
- An external CentOS EDA PDF was also visible, but it describes a VM-style EDA environment rather than the Docker entry used here.

No credential, token, password, or endpoint value was recorded.

## Container And Repository
- Container used: `nailong`
- Repository path inside container: `/workspace/VEDA`
- Repository path on host: `D:\IC_Workspace\VEDA`
- No git repository metadata was detected in this directory.

## Tool Visibility

| Tool | Result | Version / Notes |
| --- | --- | --- |
| `vcs` | FOUND | VCS script version `O-2018.09`; VCS-MX compile/run reported `O-2018.09-SP2-2_Full64` |
| `vlogan` | FOUND | VCS script version `O-2018.09`; RTL compile exit 0 |
| `verdi` | FOUND | Executable visible; not required for noninteractive Stage 1 closure |
| `dc_shell` | FOUND | DC startup reported `L-2016.03-SP1`; Stage 1 elaborate exit 0 |
| `pt_shell` | FOUND | `M-2016.12-SP1`; no STA run because there is no target library |
| `fc_shell` | NOT FOUND | Not used |
| `python` | NOT FOUND | Container default `python` is absent |
| `python3` | FOUND | Python `3.6.9` |
| `pytest` | NOT FOUND | Docker Python regression cannot run in this container without adding pytest and a Python version compatible with the repo |
| `make` | FOUND | GNU Make `4.1` |
| `gmake` | NOT FOUND | Not needed |
| `verilator` | NOT FOUND | Not used |
| `iverilog` | NOT FOUND | Not used |
| DesignWare simulation files | FOUND | Installed `dw/sim_ver` tree visible |
| DesignWare synthesis library | FOUND | `dw_foundation.sldb` visible to DC |

No license error was observed during the VCS or DC runs listed below.

## Existing Regression Results

| Command | Environment | Exit | Result |
| --- | --- | ---: | --- |
| `python -m pytest tb\model tb\unit` | Host | 0 | 26 passed |
| `python scripts\sim\run_stage1_tests.py` | Host | 0 | pytest passed, py_compile passed, RTL sim skipped because host has no VCS |
| `python scripts\lint\run_stage1_lint.py` | Host | 0 | static hygiene passed, external lint skipped |
| `make stage0-test` | Docker | 1 | failed because container has no `pytest` |
| `make stage1-test` | Docker | 1 | Python pytest unavailable and Python 3.6 cannot parse repo files that use `from __future__ import annotations`; VCS RTL substep still passed |

The Docker Python failure is an environment limitation, not an RTL simulator failure. The host Python regression remains passing.

## VCS RTL Simulation

Command:

```bash
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage1-rtl-sim'
```

Result:
- `compile_exit_code=0`
- `run_exit_code=0`
- `assertion_or_tb_errors=0`
- `pass_marker=1`
- `result=PASS`

RTL assertions were executed because `SYNTHESIS` was not defined for this run.

Modules exercised by `tb/rtl/stage1/tb_stage1_all.sv`:
- `stream_reg`
- `skid_buffer`
- `sync_fifo` with non-power-of-two depths 5 and 3
- `sram_1p_wrapper`
- `sram_2p_wrapper`
- `mul_unit`
- `add_unit`
- `mac_unit`
- `compare_max`
- `round_sat`

The testbench checks ready/valid backpressure stability, ordering, reset behavior, FIFO full/empty/wrap/pop+push behavior, SRAM read latency and read-first collision behavior, arithmetic boundary cases, round-to-nearest-even helper behavior, metadata, and `last` alignment.

## Lint And Elaboration

Command:

```bash
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage1-lint'
```

Result:
- Static RTL hygiene: PASS
- `vlogan -full64 -sverilog -work work <stage1 RTL files>`: exit 0
- vlogan diagnostics: none
- Verilator: not found, skipped

Command:

```bash
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage1-synth'
```

Result:
- `dc_shell` was found and launched.
- `analyze -format sverilog -define SYNTHESIS` completed.
- `elaborate`, `link`, and `check_design` completed for each Stage 1 top.
- `dc_exit_code=0`
- DC elaboration result: PASS

This is an analyze/elaborate/check_design synthesizability check only. `TECH_LIB_ROOT` was not set, no PDK or standard-cell target library was used, and no valid area, power, WNS, frequency, or STA conclusion exists.

## DesignWare Probe

VCS command:

```bash
docker exec nailong bash -lc 'cd /workspace/VEDA && bash scripts/sim/run_stage1_dw_probe.sh'
```

Result:
- `vcs_compile_exit_code=0`
- `vcs_run_exit_code=0`
- `probe_errors=0`
- `result=PASS`

DC command:

```bash
docker exec nailong bash -lc 'cd /workspace/VEDA/build/stage1_dw_probe && dc_shell -f /workspace/VEDA/scripts/synth/stage1_dw_probe.tcl | tee /workspace/VEDA/reports/stage_01/dw_probe_dc.log'
```

Result:
- DC analyze/elaborate/link/check_design completed for the non-conversion DW probe.
- `STAGE1_DW_DC_PROBE_PASS`
- Exit code 0.

### Probed Floating-Point Modules

| Function | Probed Module / Instance | VCS | DC | Notes |
| --- | --- | --- | --- | --- |
| FP16 multiply | `DW_fp_mult #(10, 5, 1)` | PASS | PASS | `rnd=3'b000` was used in the probe; finite normal examples matched expected FP16 bit patterns |
| FP32 add | `DW_fp_add #(23, 8, 1)` | PASS | PASS | `rnd=3'b000`; finite normal examples matched expected FP32 bit patterns |
| FP32 MAC | `DW_fp_mac #(23, 8, 1)` | PASS | PASS | `rnd=3'b000`; finite normal examples matched expected FP32 bit patterns |
| FP16 to FP32 conversion | `DW_fp_ifp_conv` plus `DW_ifp_fp_conv` probe path | COMPILES | NOT CLAIMED | Simple IEEE FP16 to FP32 bit-conversion expectation was not met; conversion semantics remain unconfirmed |
| FP32 to FP16 conversion | `DW_fp_ifp_conv` plus `DW_ifp_fp_conv` probe path | COMPILES | NOT CLAIMED | Simple IEEE FP32 to FP16 bit-conversion expectation was not met; conversion semantics remain unconfirmed |
| EXP | `DW_fp_exp #(23, 8, 1, 2)` | PASS | PASS | `exp(0.0)` matched `1.0` in the probe |
| division | `DW_fp_div #(23, 8, 1, 0)` | PASS | PASS | `4.0 / 2.0` matched `2.0` |
| reciprocal | `DW_fp_recip #(23, 8, 1, 0)` | PASS | PASS | `1 / 2.0` matched `0.5` |
| SQRT | `DW_fp_sqrt #(23, 8, 1)` | PASS | PASS | `sqrt(4.0)` matched `2.0` |
| reciprocal SQRT | `DW_fp_invsqrt #(23, 8, 1)` | PASS | PASS | `1/sqrt(4.0)` matched `0.5` |

The probe used only small finite normal values and did not validate full IEEE exception behavior, denormals, NaN, Inf, or all rounding modes. It also did not freeze latency or pipeline architecture. No DesignWare source, library, or license data was added to the repository.

## FP16 Input / FP32 Accumulation Routes

### Option A
FP16 operands -> FP16 `DW_fp_mult` product semantics -> convert product to FP32 -> FP32 accumulation.

Assessment:
- Uses the successfully probed FP16 multiply and FP32 add.
- Product is rounded at FP16 precision before accumulation.
- Needs a verified finite FP16-to-FP32 conversion boundary; the probed DW conversion path is not yet usable as-is.
- Lower arithmetic cost is likely than full FP32 multiplication, but no PPA claim can be made without a real target library.

### Option B
FP16 operands -> finite FP16-to-FP32 conversion -> FP32 multiply -> FP32 accumulation.

Assessment:
- Matches the provisional Stage 0 direction more cleanly for FP32-equivalent accumulation.
- Can use the successfully probed FP32 `DW_fp_mac` or FP32 multiply/add path.
- Requires a small, verified, synthesizable finite FP16-to-FP32 converter or a confirmed DW conversion configuration.
- Recommended as `PROPOSED`, pending user confirmation of product semantics, fused versus non-fused behavior, special-value policy, and wrapper latency.

### Option C
FP16 operands -> finite FP16-to-FP32 conversion -> `DW_fp_mac` FP32 fused/non-fused MAC wrapper.

Assessment:
- `DW_fp_mac #(23, 8, 1)` compiled, simulated, and elaborated in the probe.
- Fused versus non-fused numerical semantics must be explicitly chosen before freezing.
- Backpressure should be handled by a ready/valid wrapper with explicit pipeline stages; latency is not frozen.

## Recommendation

`PROPOSED`: use a verified finite-only FP16-to-FP32 conversion boundary followed by FP32 `DW_fp_mac` or FP32 multiply/add accumulation inside a ready/valid wrapper. Do not rely on the currently probed DW conversion path until its parameterization and expected bit behavior are resolved.

The final FP MAC interface, exact numerical semantics, fused/non-fused behavior, special-value handling, and latency remain user decisions. Stage 2 final numeric RTL remains blocked until those decisions are frozen.

## PPA Limitation

No PDK, target `.lib`, SRAM macro, memory compiler output, layout database, or P&R flow was used. The DC runs are not valid PPA. They cannot be used for area, power, WNS, frequency, or paper metric comparison.
