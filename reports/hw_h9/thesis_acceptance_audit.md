# Hardware Stage H9 Thesis Acceptance Audit

Status: PASS for undergraduate thesis scope.

Strict IP-grade verification status:

```text
STRICT IP-GRADE H9 VERIFICATION NOT CLOSED
```

## Architecture

- 8x8x2 paper PE hierarchy is implemented with 128 physical PE cells.
- QK uses inner-product mode.
- sV uses outer-product mode.
- H9 native full-array mapping is implemented.
- SFU/PE element-serial interleaving is implemented.
- Bounded score buffer and probability FIFO are implemented.
- `ATTENTION_SCHEDULE=STAGED` and `ATTENTION_SCHEDULE=INTERLEAVED` remain
  selectable for paper-array Attention.

## Numeric Correctness

H9 bit model and H8 bit model remain identical for the covered configurations:

| Metric | Result |
|---|---:|
| max_abs_error | 0 |
| MAE | 0 |
| RMSE | 0 |
| relative_L2 | 0 |
| max_ULP | 0 |
| cosine | 1 |
| attention ranking | match |

Implemented RTL configurations match the accepted reference vectors.

## Matched RTL Performance

The gain is native full-array mapping plus SFU/PE interleaving combined. It is
not a pure interleaving-only number.

| D_HEAD | Seq | Staged RTL | Interleaved RTL | Result |
|---:|---:|---:|---:|---|
| 8 | 16 | 1363 | 1169 | PASS |
| 8 | 32 | 2707 | 2209 | PASS |
| 16 | 16 | 2472 | 1171 | PASS |
| 16 | 32 | 4920 | 2211 | PASS |
| 64 | 16 | 9126 | 1183 | PASS |
| 64 | 32 | 18198 | 2223 | PASS |

## Cycle Model

`model/attention/paper_interleaved_cycle_model.py` is calibrated to the matched
RTL total-cycle interval for D_HEAD 8, 16, and 64 at seq 1, 2, 8, 16, 32, and
64. The reported total-cycle delta is 0 for all entries.

The structural non-interleaved and overlap subtotal values remain estimates
and are not used as RTL acceptance counters.

## Functional Hierarchy

- Multi-head H9 interleaved RTL: PASS for H1/D8, H2/D8, H4/D8, H2/D16, and
  H1/D64.
- Full-layer H9 interleaved RTL: PASS for H1/D8, H2/D8, H2/D8 two-token,
  H4/D8, and H2/D16.
- Long-sequence/cache-full: PASS for MAX_SEQ_LEN=32 plus one extra token.

## Transaction Semantics

- Current-token causal semantics are preserved.
- All-head atomic K/V commit is preserved.
- `valid_seq_len` changes only through successful commit.
- Cache-full extra token does not add a commit.
- Output/done transaction counts are conserved.
- The next token does not start before the previous final done.

## Reset, Backpressure, and Assertions

- Direct H9 datapath reset: 64/64 PASS.
- Direct H9 random backpressure: 20/20 fixed seeds PASS.
- Independent multi-head reset: 20/20 PASS.
- Independent multi-head random: 24/24 PASS.
- Full-layer directed reset regression: PASS.
- Full-layer external output/done backpressure: PASS.
- Stage7 RMSNorm, Residual, and FFN single-module backpressure regressions:
  PASS.
- Known deadlock/livelock/overflow/underflow: none in accepted runs.
- Stalled payload stability checks: PASS.
- Explicit H9 SVA properties: 23.
- Positive bind execution: PASS.
- Negative assertion tests: 23/23 trigger the intended property.

## Tool and Regression Baseline

- H9 host/model tests: PASS.
- H9 VCS RTL simulations: PASS.
- H9 lint/vlogan: PASS.
- H9 DC analyze/elaborate/link/check_design structural checks: PASS.
- Stage5/6/7/8 regressions: PASS.

The DC result is a structural check only. No timing closure, frequency, WNS/TNS,
area, power, PPA, STA, floorplan, placement, CTS, routing, or post-route result
is claimed.

## Deferred Strict Verification

The full internal transformer-layer reset matrix and full-layer internal
multi-endpoint random-backpressure matrix remain deferred IP-grade
verification enhancements. They are documented in
`reports/hw_h9/deferred_ip_verification.md`.

## Decision

```text
HARDWARE STAGE H9 PASS — UNDERGRADUATE THESIS SCOPE
```

```text
STRICT IP-GRADE H9 VERIFICATION NOT CLOSED
```
