# HW-H9-N1 Root Cause Report

## Root Cause

The shared `fp32_add_wrapper` used the wrong DesignWare rounding-mode encoding
for the project RNE numeric contract.

Historical H9:

```systemverilog
localparam [2:0] ROUND_NEAREST_EVEN = 3'b100;
```

Repaired HW-H9-N1:

```systemverilog
localparam [2:0] ROUND_NEAREST_EVEN = 3'b000;
```

The first M3 real Q2 W2 pair is sensitive to this encoding:

```text
3c81aa0c + 39699f40
```

The old W2 PE reduction path produced `3c837d4b`; the hardware-aware bit
model, NumPy float32, and repaired wrapper/reduction path produce `3c837d4a`.

## Required Root-Cause Questions

| Question | Answer |
|---|---|
| Concrete module/signal | `rtl/arithmetic/fp32_add_wrapper.sv`, `u_dw_fp_add.rnd` via `ROUND_NEAREST_EVEN` |
| Numeric or transaction | Numeric rounding-mode error |
| Actual DW inputs | `3c81aa0c` and `39699f40` |
| Source of `3c837d4b` | Current W2 pair add under old DW rounding-mode encoding |
| Current/old/stale/etc. | Current pair wrong rounding, not stale or misindexed |
| Why artificial vectors missed | No prior vector hit this sensitive rounding boundary in W2 reduction |
| Why Q2 triggers | Real Q2 W2 products create the exact sensitive pair at tile base 8 |
| Why H8/H9 both fail | Both use the same FFN W2 PE/reduction/add wrapper path |
| W1 affected | No observed M3 W1 mismatch before W2; protected by wrapper fix |
| Projection affected | No observed M3 projection mismatch; protected by wrapper fix |
| Attention affected | No observed M3 Attention mismatch; protected where wrapper is used |
| Only FFN W2 observed | Yes, first stable real-Q2 observed failure is FFN W2 |
| Ready/backpressure dependent | No; no-stall and output-stall Q2 runs pass after fix |
| DW latency dependent | No latency change; wrapper remains one output register stage |
| H9 cycle count affected | No external cycle count change in matched no-stall calibration |
