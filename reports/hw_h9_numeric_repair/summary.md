# HW-H9-N1 Summary

HW-H9-N1 fixes the post-acceptance real-weight numeric mismatch found by ML-M3
in the common H8/H9 FFN W2 reduction path.

## Summary

- Historical H9 tag: `hw-h9-sfu-pe-interleaving-thesis-accepted`
- Historical commit: `9e0b4c9ba42356ee68e489e99cc5cf64e94f607e`
- Repair branch: `hw/h9-real-weight-numeric-repair`
- Original mismatch: 54/64 output dimensions
- First divergent boundary: `w2_output_fp32_edge`
- First divergent operation:
  `3c81aa0c + 39699f40`, old RTL `3c837d4b`, expected `3c837d4a`
- Root cause: wrong DesignWare FP32 add RNE `rnd` encoding in shared
  `fp32_add_wrapper`
- Fix: set wrapper RNE to `3'b000`; add reduction association assertions and
  numeric regressions
- Q2 length1: H8 staged PASS, H9 interleaved PASS, 64/64 bit-exact
- Matched H9 no-stall cycle totals: unchanged
- PDK/STA/P&R/PPA: not run

## M3 Resume Criteria

M3 may resume one-token and later deployment validation after the repair branch
is committed, pushed, and tagged as:

```text
hw-h9-real-weight-numeric-repair-accepted
```

M3 should not use the old thesis tag for real Q2 deployment validation.
