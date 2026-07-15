# Hardware Stage H9 Deferred IP-Grade Verification

Status: deferred; not part of undergraduate thesis acceptance closure.

## Deferred Items

- Independent reset injection at every internal micro-stage of the real
  `transformer_layer` DUT.
- Broad full-layer internal multi-endpoint randomized backpressure across
  RMSNorm, Attention/MHA, Residual, FFN, final output, and done boundaries.
- Exhaustive simultaneous stall combinations across full-layer child engines.
- Verification-only ready gates for child-stage interfaces that are not
  legally controllable through the production top-level.
- Functional coverage closure and assertion coverage closure.
- Formal proof of H9 stream, mode, reset, transaction, and cache properties.
- IP signoff-level verification packaging.

## Why Deferred

The current production `transformer_layer` top does not expose legal testbench
control over every requested internal ready/valid boundary. Some internal
stages are only observable through top-level progress, child completion, and
bound monitors, not independently controllable as public verification
endpoints.

Closing the strict target cleanly would require additional verification
infrastructure such as:

- bind observers for internal FSM state, counters, FIFO occupancy, and
  transaction IDs;
- a test-only wrapper that can intercept selected child interfaces without
  changing production behavior;
- verification-only ready gates with explicit synthesis exclusion;
- functional coverage collection for reset, stall, mode, head, token, and
  cache scenarios;
- assertion coverage collection for the existing SVA set;
- formal bounded proofs for protocol safety and transaction conservation.

## Thesis Acceptance Impact

These tests are valuable for IP delivery, but they are outside the core
undergraduate thesis scope and time budget. The thesis conclusion depends on
the implemented architecture, bit-exact numerical behavior, matched RTL cycle
benefit, integration into the full Transformer layer, major hierarchy
reset/backpressure evidence, assertion execution, lint, DC structural checks,
and preserved Stage5/6/7/8 regressions. Those items are closed by the H9 thesis
acceptance regression.

This repository therefore may claim:

```text
HARDWARE STAGE H9 PASS — UNDERGRADUATE THESIS SCOPE
```

It must not claim:

```text
FULL IP-GRADE VERIFICATION CLOSURE
```

The strict status remains:

```text
STRICT IP-GRADE H9 VERIFICATION NOT CLOSED
```
