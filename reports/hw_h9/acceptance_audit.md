# Hardware Stage H9 Acceptance Audit

Status: not accepted.

## Closed Items

- Fixed workspace stayed at `D:/IC_Workspace/VEDA`.
- Branch stayed at `hw/h9-sfu-pe-interleaving`.
- H8 remains the accepted hardware baseline.
- Paper schedule evidence and repository design decisions are documented.
- H9 native full-array mapping model covers D_HEAD 8, 16, 64, and 128.
- Bounded score buffer and probability FIFO RTL modules exist.
- H9 Python model tests pass.
- H9 vs H8 host bit-model comparison remains bit-exact for D_HEAD 8, 16, 64,
  and 128.
- Matched single-head RTL A/B uses the same `single_head_attention` top,
  inputs, DesignWare wrappers, clock/reset, and ready environment for paper
  staged and paper interleaved schedules.
- Matched RTL seq16 and seq32 performance objective is met for D_HEAD 8, 16,
  and 64.
- Performance attribution is corrected: the gain is native full-array mapping
  plus interleaving, not pure interleaving.
- `model/attention/paper_interleaved_cycle_model.py` is calibrated to the
  matched RTL counter interval for D_HEAD 8, 16, and 64 at seq 1, 2, 8, 16,
  32, and 64.
- H9 multi-head interleaved RTL runs pass for H1/D8, H2/D8, H4/D8, H2/D16,
  and H1/D64.
- H9 full-layer interleaved RTL runs pass for H1/D8, H2/D8, H2/D8 two-token,
  H4/D8, and H2/D16.
- H9 long-sequence/cache-full RTL run passes for H1/D8, MAX_SEQ_LEN=32, plus
  one cache-full extra token.
- H9 lint/vlogan passes.
- H9 DC structural check passes for analyze/elaborate/link/check_design and
  hierarchy only.
- Stage5/6/7/8 regression commands listed in the final-closure request pass in
  the Docker EDA environment.

## Executed Commands

```text
python3 scripts/sim/run_hw_h9_tests.py
make hw-h9-test
make hw-h9-rtl-sim
make hw-h9-lint
make hw-h9-synth
make stage8-test stage8-rtl-sim stage8-lint stage8-synth
make stage7a-test
make stage7b-test stage7b-rtl-sim stage7b-lint stage7b-synth
make stage7c-test stage7c-rtl-sim stage7c-lint stage7c-synth
make stage7d-test stage7d-rtl-sim stage7d-lint stage7d-synth
make stage6-test stage6-rtl-sim stage6-lint stage6-synth
make stage5-test stage5-rtl-sim stage5-lint stage5-synth
```

All returned PASS.

## Open Items Blocking Hardware Stage H9 PASS

- Full reset interrupt matrix is not implemented as independent injection
  points for every requested micro-stage. Current coverage includes reset
  during Stage 5 provisional append and attention start, plus ordinary reset
  before H9 single-head/buffer/layer runs.
- Broad random backpressure with at least 20 fixed seeds is not implemented or
  executed. Current coverage is no external backpressure plus deterministic
  output/done backpressure.
- Assertion execution is not sufficient for final acceptance. H9 VCS commands
  compile with `-assert svaext` and pass with zero assertion failures, but the
  requested matrix still lacks negative/bind evidence and several requested
  names are represented by scoreboard checks rather than explicit named
  assertion properties.
- No H9 accepted tag has been created.

## Decision

Do not write `HARDWARE STAGE H9 PASS`.

Do not create `hw-h9-sfu-pe-interleaving-accepted`.

Do not enter Hardware Stage H10.

Stage 8 remains the accepted hardware baseline until the remaining reset,
random-backpressure, and assertion-execution conditions are implemented,
executed, and recorded.
