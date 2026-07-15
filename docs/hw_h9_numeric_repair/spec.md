# HW-H9-N1 Post-Acceptance Real-Weight Numeric Repair

## Scope

HW-H9-N1 repairs a real-weight numeric mismatch found after the historical
Hardware H9 undergraduate thesis acceptance tag:

- Historical tag: `hw-h9-sfu-pe-interleaving-thesis-accepted`
- Historical commit: `9e0b4c9ba42356ee68e489e99cc5cf64e94f607e`
- Repair branch: `hw/h9-real-weight-numeric-repair`

The old tag remains the historical thesis architecture acceptance baseline.
It is not moved, replaced, or rewritten. The repair branch supersedes that tag
only for ML-M3 real-weight deployment validation.

## Frozen Reproduction

M3 real Q2 one-token validation found a common H8/H9 FFN W2 mismatch:

- Configuration: `N_HEAD=8`, `D_HEAD=8`, `D_MODEL=64`, `D_FFN=256`,
  `MAX_SEQ_LEN=128`, `ATTENTION_PE_ARCH=PAPER_ARRAY`
- Token/dimension: token 0, dimension 1
- RTL/H8/H9 value: `32'h3d4a2576`
- Hardware-aware expected: `32'h3d4a2572`
- Full output mismatch count: 54/64 dimensions

The first stable matching boundary is before FFN W2:

- `residual1_fp32_edge`: match
- `norm2_output_fp16_edge`: match
- `w2_output_fp32_edge`: mismatch
- `residual2_final_fp32_edge`: mismatch

The first divergent arithmetic operation is an FFN W2 reduction add:

- Cycle: 185551 in the staged trace
- Tile base: 8
- Width: 8
- Pair: 3
- Operand A: `32'h3c81aa0c`
- Operand B: `32'h39699f40`
- Old RTL result: `32'h3c837d4b`
- Expected RNE result: `32'h3c837d4a`

## Numeric Contract

All project FP32 additions use DesignWare `DW_fp_add` with round-to-nearest,
ties-to-even (RNE). For this DesignWare version the RNE `rnd` encoding used by
the project must be `3'b000`.

The repair therefore changes `rtl/arithmetic/fp32_add_wrapper.sv` so the
single shared FP32 add wrapper drives `rnd=3'b000`. The wrapper now also has an
elaboration-time guard that fails if the local RNE constant changes.

No external interface, tensor format, bit width, data ordering, KV layout,
Transformer structure, or ready/valid transaction semantics change.

## Acceptance Conditions

The repair is accepted only when:

- Known operand add replay returns `32'h3c837d4a`.
- The same operand is exercised through the real reduction and PE core path.
- Random fixed-seed reductions are bit-exact against the hardware reference.
- H8 staged and H9 interleaved full `transformer_layer` real Q2 length1
  outputs are 64/64 bit-exact against the M3 hardware-aware expected file.
- Stage5/6/7/8/H9 regressions, lint, and DC structural checks pass.
- The matched H9 cycle-model calibration stays at total-cycle delta 0.
- No PDK, STA, P&R, power, layout, or PPA claim is made.
