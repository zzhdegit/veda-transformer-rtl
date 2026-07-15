# Hardware Stage H9 Summary

Status: final verification acceptance closure checkpoint, not accepted.

## Executed In This Closure Turn

Work stayed in `D:/IC_Workspace/VEDA` on branch
`hw/h9-sfu-pe-interleaving`. The Docker EDA environment `nailong` was used for
RTL, vlogan, and DC structural checks.

The unified command completed with exit code 0:

```text
make PYTHON=python3 hw-h9-final-acceptance
```

That target executed H9 model/RTL/lint/DC, the new reset/random/assertion
closures, and Stage5/6/7/8 regression targets. No PDK, SRAM macro, STA, P&R,
timing closure, area, power, frequency, WNS, TNS, or PPA flow was run.

## H9 Results

- H9 host/model tests: PASS.
- H9 vs H8 bit-model comparison: bit-exact for D_HEAD 8, 16, 64, and 128.
- H9 calibrated cycle model: exact total-cycle match against matched RTL A/B
  for D_HEAD 8, 16, and 64 at seq 1, 2, 8, 16, 32, and 64.
- H9 matched single-head RTL A/B performance: PASS at seq16 and seq32 for
  D_HEAD 8, 16, and 64.
- H9 multi-head interleaved RTL: PASS for H1/D8, H2/D8, H4/D8, H2/D16, and
  H1/D64 through the existing Stage 5 wrapper.
- H9 full-layer interleaved RTL: PASS for H1/D8, H2/D8, H2/D8 two-token,
  H4/D8, and H2/D16.
- H9 long-sequence/cache-full: PASS for MAX_SEQ_LEN=32 plus one extra token.
- H9 lint/vlogan: PASS with only accepted DesignWare pragma-no-effect warnings.
- H9 DC structural check: PASS for analyze/elaborate/link/check_design and
  hierarchy only.
- Stage5/6/7/8 regressions in the Docker EDA environment: PASS.

Matched RTL A/B remains the performance authority. The old structural cycle
model is retained only as trend evidence and is not used to decide H9 speedup.
The measured performance gain is the combined result of native full-array
mapping plus SFU/PE interleaving; it is not pure interleaving benefit.

## Final Blocker Work

- Reset matrix harness: 64 named injection labels executed and passed against
  the H9 interleaved attention datapath with clean two-transaction recovery.
- Broad random backpressure harness: 20 fixed seeds executed and passed for
  H1 direct H9 datapath configurations across D_HEAD 8, 16, and 64.
- Assertion execution: 23 explicit named SVA properties compile, bind into
  `paper_interleaved_attention_datapath`, pass positive execution, and each
  has an isolated negative test that triggers the target property as expected.
- Strict multi-head reset: 20/20 independent reset injection rows PASS against
  the real `multi_head_generation_engine` hierarchy.
- Strict multi-head random: 24/24 fixed-seed runs PASS against the real
  `multi_head_generation_engine` hierarchy across H2/D8 seq2, H2/D8 seq8,
  H2/D8 seq16, H4/D8 seq8, H2/D16 seq8, and H1/D64 seq8.

## Remaining Acceptance Gap

Hardware Stage H9 is still not accepted. The new reset and random tests close
substantial H9 datapath coverage, the assertion blocker is closed, and the
strict independent multi-head reset/random blocker is now closed. The remaining
gap is full-layer independent reset/random endpoint coverage against the real
`transformer_layer` DUT. Several requested child-stage stall points are not
legally controllable through the current production top-level; they still need a
test-only wrapper, observation interface, or legal child-interface interception
strategy. They must not be replaced with direct datapath proxy labels, fixed
cycle waits, or forced internal ready signals.

## Acceptance Status

```text
HW-H9 IN PROGRESS, NOT ACCEPTED
```

No accepted tag was created. Stage 8 remains the accepted hardware baseline.
