# HW-H9-N1 Regression Plan

## Committed Numeric Regression

Target:

```bash
make hw-h9-numeric-repair
```

Coverage:

- known operand add replay:
  `3c81aa0c + 39699f40 -> 3c837d4a`;
- real `fp32_reduction_tree` replay;
- real `reconfigurable_pe_core` reduction replay;
- pair 0 through pair 3 directed placements;
- consecutive tile accumulation with base0/base8 W2-style vectors;
- valid gaps;
- output stalls;
- reset during reduction;
- signed zero, subnormal/FTZ-adjacent, small-plus-large, cancellation, positive
  and negative accumulation cases;
- 100 fixed-seed random reductions.

## External Real Q2 Regression

Target:

```bash
make hw-h9-q2-length1
```

The target reads the M3 artifact vector from:

```text
D:/IC_Workspace/VEDA_artifacts/ml_m3/vectors/len_1/case_len_1.mem
```

The vector is read-only and is not committed. Coverage:

- H8 staged schedule, no output stall;
- H8 staged schedule, output stall;
- H9 interleaved schedule, no output stall;
- H9 interleaved schedule, output stall;
- full `transformer_layer`;
- 64 output dimensions;
- done/commit transaction count;
- `valid_seq_len=1`.

## Broader Regression

The closing run includes:

```bash
make stage7c-test stage7c-rtl-sim stage7c-lint stage7c-synth
make stage7d-test stage7d-rtl-sim stage7d-lint stage7d-synth
make stage8-test stage8-rtl-sim stage8-lint stage8-synth
make hw-h9-test hw-h9-rtl-sim hw-h9-lint hw-h9-synth
make hw-h9-thesis-acceptance
make stage5-test stage5-rtl-sim stage5-lint stage5-synth
make stage6-test stage6-rtl-sim stage6-lint stage6-synth
make stage7a-test stage7b-test stage7b-rtl-sim stage7b-lint stage7b-synth
```

Stage2 RTL/lint/DC structural checks are also run as an extra PE sanity check.
`stage2-test` host py_compile is limited by the Docker Python 3.6.9 runtime
and pre-existing `from __future__ import annotations` usage; Stage2 RTL PE
simulation itself passes.
