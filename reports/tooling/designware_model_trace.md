# DesignWare Floating-Point Model Trace

Date: 2026-07-12

Scope: trace the Synopsys DesignWare floating-point simulation models used by the current VEDA RTL wrappers and document how data enters and leaves those wrappers. This report does not copy DesignWare source, modify RTL, or claim PDK-backed PPA.

## Environment Summary

- Docker container used: `nailong`.
- Repository path inside container: `/workspace/VEDA`.
- `SYNOPSYS`: `/usr/synopsys`.
- `VCS_HOME`: `/usr/synopsys/vc_static-O-2018.09-SP2-2/vcs-mx/`.
- `DW_SIM_DIR`: not set in the container environment.
- `DW_FOUNDATION_SLDB`: not set in the container environment.
- License-related environment variables were treated as sensitive and not recorded.

The project scripts auto-detect DesignWare locations when the environment variables are not set. The observed simulation model directory is:

```text
/usr/synopsys/dc-L-2016.03-SP1/dw/sim_ver
```

The observed Design Compiler synthetic library path used by the project runners is:

```text
/usr/synopsys/dc-L-2016.03-SP1/libraries/syn/dw_foundation.sldb
```

## Simulation Model Source

The Stage 1B, Stage 2, and Stage 3 VCS runners pass DesignWare simulation files directly to `vcs` as source files. They do not rely on `-v`, `-y`, `+libext`, `synopsys_sim.setup`, or a precompiled DesignWare simulation library for the `DW_fp_*` modules checked here.

Observed runner behavior:

- `scripts/sim/run_stage1b_vcs.sh` compiles `DW_fp_add.v` and `DW_fp_mac.v`, plus dependencies, from the detected `dw/sim_ver` directory.
- `scripts/sim/run_stage2_vcs.sh` uses the same direct source-file method for `DW_fp_add` and `DW_fp_mac`.
- `scripts/sim/run_stage3_vcs.sh` adds `DW_exp2.v`, `DW_fp_exp.v`, and `DW_fp_div.v` for exp and reciprocal support.
- Compile logs under `build/stage1b_rtl_sim`, `build/stage2_rtl_sim`, and `build/stage3_rtl_sim` show `Parsing design file` entries for these `/usr/synopsys/.../dw/sim_ver/DW_fp_*.v` files.
- No `synopsys_sim.setup` file was found in the repository tree.

Representative source list fragments:

```text
+incdir+/usr/synopsys/dc-L-2016.03-SP1/dw/sim_ver
/usr/synopsys/dc-L-2016.03-SP1/dw/sim_ver/DW_fp_add.v
/usr/synopsys/dc-L-2016.03-SP1/dw/sim_ver/DW_fp_mac.v
/usr/synopsys/dc-L-2016.03-SP1/dw/sim_ver/DW_fp_exp.v
/usr/synopsys/dc-L-2016.03-SP1/dw/sim_ver/DW_fp_div.v
```

The checked files are ASCII text `.v` files with visible module declarations:

```text
DW_fp_mac.v: module DW_fp_mac (a, b, c, rnd, z, status);
DW_fp_add.v: module DW_fp_add (a, b, rnd, z, status);
DW_fp_exp.v: module DW_fp_exp (a, z, status);
DW_fp_div.v: module DW_fp_div (a, b, rnd, z, status);
```

No `pragma protect` marker was found in the four core files above. The source is readable on this machine, but it is licensed Synopsys DesignWare material and must not be copied into this repository or external reports.

## Module Summary

| DW module | project wrapper | simulation source/library | model form | readable | synthesis binding |
|---|---|---|---|---|---|
| `DW_fp_mac #(23, 8, 1)` | `rtl/arithmetic/fp32_mac_wrapper.sv` | `/usr/synopsys/dc-L-2016.03-SP1/dw/sim_ver/DW_fp_mac.v` plus dependencies | Direct ASCII Verilog simulation source passed to VCS | Yes, licensed material | `dw_foundation.sldb` through `synthetic_library` and `link_library` |
| `DW_fp_add #(23, 8, 1)` | `rtl/arithmetic/fp32_add_wrapper.sv` | `/usr/synopsys/dc-L-2016.03-SP1/dw/sim_ver/DW_fp_add.v` plus dependencies | Direct ASCII Verilog simulation source passed to VCS | Yes, licensed material | `dw_foundation.sldb` through `synthetic_library` and `link_library` |
| `DW_fp_exp #(23, 8, 1, 2)` | `rtl/arithmetic/fp32_exp_wrapper.sv` | `/usr/synopsys/dc-L-2016.03-SP1/dw/sim_ver/DW_fp_exp.v` plus `DW_exp2.v` and dependencies | Direct ASCII Verilog simulation source passed to VCS | Yes, licensed material | `dw_foundation.sldb` through `synthetic_library` and `link_library` |
| `DW_fp_div #(23, 8, 1, 0)` | `rtl/arithmetic/fp32_recip_wrapper.sv` | `/usr/synopsys/dc-L-2016.03-SP1/dw/sim_ver/DW_fp_div.v` plus dependencies | Direct ASCII Verilog simulation source passed to VCS | Yes, licensed material | `dw_foundation.sldb` through `synthetic_library` and `link_library` |

## VCS Resolution Mechanism

The floating-point DW modules are resolved at VCS compile/elaboration from explicit source-file arguments:

- Uses direct paths such as `$DW_SIM_DIR_DETECTED/DW_fp_mac.v`.
- Uses `+incdir+$DW_SIM_DIR_DETECTED`.
- Does not use `-v` for these DW model files.
- Does not use `-y` or `+libext` for these DW model files.
- Does not use a project-local `synopsys_sim.setup`.
- Does not use a precompiled DesignWare library for these checked modules.

Any `-L` observed in generated VCS build internals is for the VCS runtime/link environment, not for resolving these DesignWare Verilog modules.

## DC Binding

The Stage 1B, Stage 2, Stage 3, and Stage 4 DC Tcl scripts use the same binding pattern:

```tcl
set dw_sldb $::env(DW_FOUNDATION_SLDB)
set synthetic_library [list $dw_sldb]
set link_library [concat "*" $synthetic_library]
analyze -format sverilog -define SYNTHESIS $rtl_files
elaborate ...
link
check_design
```

The Python synth runners auto-detect `dw_foundation.sldb` when `DW_FOUNDATION_SLDB` is not set and pass it to `dc_shell` through the environment.

`dw_foundation.sldb` is the DesignWare synthetic library used by Design Compiler to recognize and elaborate/synthesize DW components. It is not a PDK, standard-cell target library, SRAM macro, or timing library. With no target library in this project state, DC results are elaboration/synthesizability checks only. They are not mapped-gate netlists, not valid STA, and not valid area, frequency, power, or WNS data.

## Wrapper Data Flow

### `fp32_mac_wrapper`

Text data flow:

```text
in_valid/in_ready
-> project finite-input check on in_a, in_b, in_c
-> project sanitation to zero when invalid
-> DW_fp_mac(a=dw_a, b=dw_b, c=dw_c, rnd=3'b100)
-> DW z/status
-> project stream_reg packs {invalid, status, result} with meta/last
-> out_valid/out_ready/out_result/out_status/out_invalid/out_meta/out_last
```

Mapping:

- DW operation: `z = a * b + c`.
- Rounding input: project constant `3'b100`, documented by Stage 1B as round-to-nearest-even for the local DW model.
- DW status is preserved in `out_status`.
- Project logic owns input invalid detection, invalid sanitation, metadata, `last`, and ready/valid holding.
- The DW instance has no `clk` port in the current simulation model declaration.

### `fp32_add_wrapper`

Text data flow:

```text
in_valid/in_ready
-> project finite-input check on in_a and in_b
-> project sanitation to zero when invalid
-> DW_fp_add(a=dw_a, b=dw_b, rnd=3'b100)
-> DW z/status
-> project stream_reg packs {invalid, status, result} with meta/last
-> out_valid/out_ready/out_result/out_status/out_invalid/out_meta/out_last
```

Mapping:

- DW operation: `z = a + b`.
- Rounding input: `3'b100`.
- DW status is preserved in `out_status`.
- Project logic owns invalid detection, sanitation, metadata, `last`, and ready/valid holding.
- The DW instance has no `clk` port in the current simulation model declaration.

### `fp32_exp_wrapper`

Text data flow:

```text
in_valid/in_ready
-> project finite-input check
-> project low-input clamp check against -20.0
-> project sanitation to zero for invalid or clamped inputs
-> DW_fp_exp(a=dw_a)
-> DW z/status
-> project output selection for invalid/clamp/DW result
-> project stream_reg packs {invalid, status, result} with meta/last
-> out_valid/out_ready/out_result/out_status/out_invalid/out_meta/out_last
```

Mapping:

- DW operation: `z = exp(a)` for inputs that are neither invalid nor clamped.
- DW parameters include `arch=2`.
- There is no `rnd` port on the observed `DW_fp_exp` module declaration.
- Project logic owns nonfinite invalid detection, low-value clamp to zero, metadata, `last`, and ready/valid holding.
- The DW instance has no `clk` port in the current simulation model declaration.

### `fp32_recip_wrapper`

Text data flow:

```text
in_valid/in_ready
-> project finite and nonzero input check
-> project denominator sanitation to 1.0 when invalid
-> DW_fp_div(a=1.0, b=dw_den, rnd=3'b100)
-> DW z/status
-> project output selection for invalid/DW result
-> project stream_reg packs {invalid, status, result} with meta/last
-> out_valid/out_ready/out_result/out_status/out_invalid/out_meta/out_last
```

Mapping:

- The reciprocal wrapper does not instantiate a DW reciprocal module.
- It implements reciprocal as `1.0 / x` through `DW_fp_div`.
- DW operation in this wrapper: `z = a / b`, with `a = 32'h3F80_0000` and `b = in_a` for valid inputs.
- Rounding input: `3'b100`.
- Project logic owns zero/nonfinite invalid detection, invalid result forcing, metadata, `last`, and ready/valid holding.
- The DW instance has no `clk` port in the current simulation model declaration.

## Latency Source

The current project wrapper latency of 1 cycle is an external wrapper latency, not an internal DesignWare pipeline latency.

Evidence:

- The observed module declarations for `DW_fp_mac`, `DW_fp_add`, `DW_fp_exp`, and `DW_fp_div` have no `clk` port.
- Each project wrapper feeds the DW combinational result/status into a project `stream_reg`.
- During `out_valid && !out_ready`, the `stream_reg` holds the already registered payload, status, invalid flag, metadata, and `last` stable.
- The DW combinational outputs may change as upstream inputs change, but stalled visible outputs are held by the project register.

Therefore, the current wrappers are correctness baselines around combinational DW arithmetic plus one visible output register. They should not be described as internally pipelined DW datapaths or high-frequency physical implementations.

## Equivalence and Limitations

- VCS simulation does include executable DesignWare models for the checked DW floating-point modules.
- On this machine, those models are direct readable ASCII Verilog simulation files under the Synopsys installation.
- DC uses `dw_foundation.sldb` to recognize and elaborate/synthesize DW operators, not the VCS `.v` simulation files.
- The simulation models and synthetic-library binding are intended to represent the same DesignWare functions, but this project has not performed gate-level equivalence or PDK-backed mapped synthesis for these operators.
- No result in this investigation is formal PPA, STA, or final gate-level signoff.
- The Synopsys `.v` simulation models, `.sldb` files, documentation, installation contents, and license configuration are commercial licensed assets and must remain outside Git.
