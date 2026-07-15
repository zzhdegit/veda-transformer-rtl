# HW-H9-N1 Root Cause

## Classification

The HW-H9-N1 failure is a numeric rounding-mode contract bug in the shared
FP32 add wrapper, not a transaction association bug.

The first divergent operation uses the real operands:

- A = `32'h3c81aa0c`
- B = `32'h39699f40`

Those operands are the values consumed by the FFN W2 reduction tree add for
tile base 8, width 8, pair 3. The wrong result `32'h3c837d4b` is the old
DesignWare add result for that same current pair when the wrapper's rounding
mode is not the project RNE mode. It is not the previous pair, next pair, old
pipeline result, lane-mask residue, tile residue, dropped partial sum, or
duplicated partial sum.

## Defect

In the historical H9 tag, `rtl/arithmetic/fp32_add_wrapper.sv` defined:

```systemverilog
localparam [2:0] ROUND_NEAREST_EVEN = 3'b100;
```

For the DesignWare `DW_fp_add` used by this project, the project RNE contract
must use:

```systemverilog
localparam [2:0] ROUND_NEAREST_EVEN = 3'b000;
```

The W2 operand pair is near a rounding boundary, so the wrong rounding-mode
encoding changes the low bit from `...4a` to `...4b`. The independent NumPy
float32 result, hardware-aware bit model, and fixed wrapper replay all agree
on `32'h3c837d4a`.

## Why Earlier Regressions Missed It

The old hand-authored vectors did not include a sensitive W2 reduction pair
whose exact sum distinguishes the incorrect DesignWare `rnd` encoding from
project RNE. Most prior add/reduction vectors were far from a tie-like
rounding boundary or only checked broader integration behavior.

The M3 real Q2 weights are stable reproducers because their W2 lane products
create the tile-base-8 pair:

```text
3c81aa0c + 39699f40
```

That pair exposes the one-bit rounding difference before the W2 tile sum is
accumulated into the final FFN result.

## H8/H9 Scope

H8 staged and H9 interleaved both use the same Stage7 FFN W2 path and the same
`reconfigurable_pe_core`/`fp32_reduction_tree`/`fp32_add_wrapper` arithmetic.
Therefore both schedules produced the same wrong value. This is why the issue
is not an H9 schedule divergence.

Potentially affected users are all users of `fp32_add_wrapper` when they hit a
similarly sensitive operand pair. In the M3 trace, the first observed stable
failure is FFN W2. W1, Projection, and Attention did not show an observed M3
boundary mismatch before W2 in the one-token artifact, but they are protected
by the same wrapper-level fix and regressions.
