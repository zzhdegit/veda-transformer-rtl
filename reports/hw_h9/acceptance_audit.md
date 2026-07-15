# Hardware Stage H9 Acceptance Audit

Status:

```text
HARDWARE STAGE H9 PASS — UNDERGRADUATE THESIS SCOPE
```

Strict verification status:

```text
STRICT IP-GRADE H9 VERIFICATION NOT CLOSED
```

## Acceptance Basis

Hardware Stage H9 is accepted as an undergraduate thesis baseline. The accepted
scope is defined in `docs/hw_h9/thesis_acceptance_scope.md`. This audit does
not claim full IP-grade verification closure.

## Closed Thesis Items

- Fixed workspace stayed at `D:/IC_Workspace/VEDA`.
- Branch stayed at `hw/h9-sfu-pe-interleaving`.
- H9 implements an 8x8x2 paper PE hierarchy with 128 physical PE cells.
- QK uses inner-product mode and sV uses outer-product mode.
- H9 native full-array mapping is implemented.
- SFU/PE element-serial interleaving is implemented.
- Bounded score buffer and probability FIFO RTL modules exist.
- Paper staged and paper interleaved schedule selections remain available.
- H9 Python model tests pass.
- H9 vs H8 bit-model comparison remains bit-exact for D_HEAD 8, 16, 64, and
  128.
- Matched single-head RTL A/B uses the same `single_head_attention` top,
  inputs, DesignWare wrappers, clock/reset, and ready environment for paper
  staged and paper interleaved schedules.
- Matched RTL seq16 and seq32 performance objective is met for D_HEAD 8, 16,
  and 64.
- Performance attribution is corrected: the gain is native full-array mapping
  plus interleaving, not pure interleaving.
- `model/attention/paper_interleaved_cycle_model.py` is calibrated to the
  matched RTL counter interval for D_HEAD 8, 16, and 64 at seq 1, 2, 8, 16,
  32, and 64 with total-cycle delta 0.
- H9 multi-head interleaved RTL runs pass for H1/D8, H2/D8, H4/D8, H2/D16,
  and H1/D64.
- H9 full-layer interleaved RTL runs pass for H1/D8, H2/D8, H2/D8 two-token,
  H4/D8, and H2/D16.
- H9 long-sequence/cache-full RTL run passes for H1/D8, MAX_SEQ_LEN=32, plus
  one cache-full extra token.
- Current-token causal semantics, all-head atomic K/V commit, cache-full, and
  valid_seq_len semantics remain unchanged.
- Direct H9 datapath reset matrix passes 64/64 rows.
- Direct H9 datapath random backpressure passes 20/20 fixed seeds.
- Independent multi-head reset passes 20/20 rows against the real
  `multi_head_generation_engine` hierarchy.
- Independent multi-head random backpressure passes 24/24 fixed-seed runs
  against the real `multi_head_generation_engine` hierarchy.
- Assertion execution matrix contains 23 explicit named SVA properties with
  positive bind execution and 23/23 isolated negative tests triggering the
  intended properties.
- H9 lint/vlogan passes.
- H9 DC structural check passes for analyze/elaborate/link/check_design and
  hierarchy only.
- Stage5/6/7/8 regression commands pass in the Docker EDA environment.

## Thesis Regression Command

```text
make PYTHON=python3 hw-h9-thesis-acceptance
```

The generated record is `reports/hw_h9/thesis_acceptance_regression.md`.

## Deferred Strict Items

The following strict IP-grade verification items remain open and are explicitly
not claimed as complete:

- full internal `transformer_layer` reset injection at every requested
  micro-stage;
- broad full-layer internal multi-endpoint random backpressure across all
  child-stage ready/valid boundaries;
- exhaustive simultaneous full-layer internal stall coverage;
- functional coverage closure;
- assertion coverage closure;
- formal proof;
- IP signoff-level verification.

These items are valuable for IP delivery, but they are deferred from the
undergraduate thesis acceptance. Details are recorded in
`reports/hw_h9/deferred_ip_verification.md`.

## Decision

```text
HARDWARE STAGE H9 PASS — UNDERGRADUATE THESIS SCOPE
```

Do not write or imply:

```text
FULL IP-GRADE VERIFICATION CLOSURE
```

No PDK, SRAM macro binding, STA, P&R, timing closure, area, power, frequency,
WNS/TNS, or PPA flow was run or claimed.
