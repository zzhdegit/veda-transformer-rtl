# HW-H9-N1 Full Regression

## Numeric and Real Q2

| Command | Result |
|---|---|
| `make hw-h9-numeric-repair` | PASS |
| `make hw-h9-q2-length1` | PASS |

## Stage7

| Command group | Result |
|---|---|
| `make stage7c-test stage7c-rtl-sim stage7c-lint stage7c-synth` | PASS |
| `make stage7d-test stage7d-rtl-sim stage7d-lint stage7d-synth` | PASS |

## Stage8/H9

| Command group | Result |
|---|---|
| `make stage8-test stage8-rtl-sim stage8-lint stage8-synth` | PASS |
| `make hw-h9-test hw-h9-rtl-sim hw-h9-lint hw-h9-synth hw-h9-thesis-acceptance` | PASS |

The H9 thesis acceptance bundle also reran and passed:

- H9 model tests;
- H9 RTL bundle;
- H9 cycle calibration;
- direct reset;
- direct random backpressure;
- multi-head reset;
- multi-head random backpressure;
- H9 assertions;
- H9 lint;
- H9 DC structural;
- Stage8 regression;
- Stage7 regression;
- Stage6 regression;
- Stage5 regression.

## Stage2 Extra PE Sanity

Additional PE sanity checks:

| Command | Result |
|---|---|
| `make stage2-rtl-sim stage2-lint stage2-synth` | PASS |
| `make stage2-test` | RTL sub-run PASS; host py_compile blocked by Python 3.6.9 `from __future__ import annotations` support |

The Stage2 host py_compile issue is pre-existing environment compatibility and
is not an HW-H9-N1 RTL numeric failure.

## Lint and DC

All required Stage5/6/7/8/H9 lint and DC structural checks passed. DC checks
were analyze/elaborate/link/check_design/hierarchy-only. No PDK, STA, P&R,
layout, power, or PPA step was run.
