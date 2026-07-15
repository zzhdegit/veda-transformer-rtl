# Hardware Stage H9 Summary

Status: final-closure regression checkpoint, not accepted.

## Executed In This Closure Turn

Work stayed in `D:/IC_Workspace/VEDA` on branch
`hw/h9-sfu-pe-interleaving`.

The Docker EDA environment `nailong` was used for RTL, vlogan, and DC checks.
The following commands completed:

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

All commands above returned PASS.

## H9 Results

- H9 host/model tests: PASS.
- H9 vs H8 bit-model comparison: bit-exact for D_HEAD 8, 16, 64, and 128.
- H9 calibrated cycle model: exact total-cycle match against matched RTL A/B
  for D_HEAD 8, 16, and 64 at seq 1, 2, 8, 16, 32, and 64.
- H9 RTL simulation: PASS.
- H9 lint/vlogan: PASS with only accepted DesignWare pragma-no-effect warnings.
- H9 DC structural check: PASS for analyze/elaborate/link/check_design and
  hierarchy only. No PPA is claimed.

Matched RTL A/B remains the performance authority. The old structural cycle
model is retained only as trend evidence and is not used to decide H9 speedup.

The measured performance gain is the combined result of native full-array
mapping plus SFU/PE interleaving. It must not be described as pure interleaving
benefit.

## Covered H9 RTL Configurations

- Single-head smoke: D_HEAD 8, 16, and 64.
- Matched single-head A/B: staged and interleaved schedules at D_HEAD 8, 16,
  and 64 for seq 1, 2, 8, 16, 32, and 64.
- Deterministic output/done backpressure matched A/B: D_HEAD 8, 16, and 64 at
  seq16 and seq32.
- Multi-head interleaved Stage 5 wrapper: H1/D8, H2/D8, H4/D8, H2/D16, and
  H1/D64, each through MAX_SEQ_LEN=8 plus one cache-full extra token.
- Long sequence/cache-full: H1/D8 through MAX_SEQ_LEN=32 plus one extra token.
- Full `transformer_layer` interleaved schedule: H1/D8, H2/D8, H2/D8
  two-token, H4/D8, and H2/D16.

## Remaining Acceptance Blockers

Hardware Stage H9 remains not accepted because these final exit conditions are
not yet fully closed:

- the requested reset interrupt matrix is only partially covered;
- the requested broad deterministic/random backpressure matrix with at least
  20 fixed random seeds is not implemented or executed;
- assertions are compiled and executed in the passing H9 VCS runs, but the
  requested assertion execution matrix is still incomplete because it lacks
  negative/bind evidence and several named checks are implemented as testbench
  scoreboards rather than explicit named SV assertions.

## Acceptance Status

Hardware Stage H9 remains:

```text
HW-H9 IN PROGRESS, NOT ACCEPTED
```

No accepted tag was created. Stage 8 remains the accepted hardware baseline.
