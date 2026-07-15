# Hardware Stage H9 Undergraduate Thesis Acceptance Scope

Status: scope frozen for undergraduate thesis acceptance.

This document defines the acceptance level used for Hardware Stage H9 after
the strict IP-grade full-layer internal stress closure was deferred. It does
not erase the strict verification gap. It separates the undergraduate thesis
baseline from a later IP delivery verification target.

## Accepted Thesis Scope

Hardware Stage H9 is accepted for undergraduate thesis scope when the
repository demonstrates:

- the paper Attention architecture is implemented with an 8x8x2 array and
  128 physical paper PE cells;
- QK uses inner-product mode and sV uses outer-product mode;
- native full-array dimension mapping is implemented;
- SFU/PE element-serial interleaving is implemented with bounded score and
  probability buffers;
- staged and interleaved paper schedules are selectable;
- H9 bit-model outputs match H8 bit-model outputs with zero numerical error;
- implemented RTL configurations match the accepted reference vectors;
- the interleaved path has measured matched RTL cycle improvement at seq16 and
  seq32 for D_HEAD 8, 16, and 64;
- the cycle model is calibrated to the matched RTL total-cycle interval;
- multi-head and full-layer integration tests pass for the documented H9
  configurations;
- current-token causal semantics, all-head atomic K/V commit, cache-full, and
  valid_seq_len semantics are preserved;
- direct H9 datapath reset/random stress, independent multi-head reset/random
  stress, and assertion positive/negative execution pass;
- H9 lint/vlogan, DC structural checks, and Stage5/6/7/8 regressions pass;
- no PDK, STA, P&R, timing closure, area, power, frequency, WNS/TNS, or PPA
  claim is made.

The thesis acceptance status is:

```text
HARDWARE STAGE H9 PASS — UNDERGRADUATE THESIS SCOPE
```

## Strict IP-Grade Verification Scope

The following items are valuable for IP handoff and production-quality
verification, but are not mandatory blockers for the undergraduate thesis
acceptance level:

- independent reset injection at every true internal micro-stage of the real
  `transformer_layer` DUT;
- fully controllable randomized backpressure on RMSNorm, MHA, Residual, FFN,
  and all full-layer internal interfaces;
- exhaustive full-layer internal multi-endpoint simultaneous stall coverage;
- verification-only ready gates or wrappers for all child-stage boundaries;
- production RTL observability comparable to a DFT-style verification shell;
- functional coverage closure;
- assertion coverage closure;
- formal property proof;
- IP signoff-level verification closure.

The strict status remains:

```text
STRICT IP-GRADE H9 VERIFICATION NOT CLOSED
```

## Boundary

The deferred strict items must not be relabeled as complete. They are not
unnecessary; they are useful for later IP delivery. They are deferred because
they require additional verification infrastructure around the production
`transformer_layer` top, while the undergraduate thesis research question is
the paper Attention full-array mapping, SFU/PE interleaving architecture,
numeric correctness, integration feasibility, and reproducible system-level
baseline.
