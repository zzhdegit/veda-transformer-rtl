# Hardware Stage H9 Final Acceptance Regression

Status: regression bundle PASS, Hardware Stage H9 not accepted.

## Environment

- Workspace: `D:/IC_Workspace/VEDA`
- Docker workspace: `/workspace/VEDA`
- Branch: `hw/h9-sfu-pe-interleaving`
- Start HEAD for this closure turn: `1a696a3`
- Docker EDA environment: `nailong`
- Python override: `PYTHON=python3`

## Command

```text
docker exec nailong bash -lc "cd /workspace/VEDA && make PYTHON=python3 hw-h9-final-acceptance"
```

Result: PASS, exit code 0.

## Included Targets

```text
python3 scripts/sim/run_hw_h9_tests.py
make hw-h9-test
make hw-h9-rtl-sim
make hw-h9-reset-test
make hw-h9-random-backpressure-test
make hw-h9-assertion-test
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

All targets returned PASS in the Docker run.

## New Final-Closure Results

| Area | Result | Evidence |
|---|---|---|
| Reset matrix | PASS for direct H9 datapath, 64/64 labels | `reports/hw_h9/reset_execution_matrix.md` |
| Random backpressure | PASS for direct H9 datapath, 20/20 seeds | `reports/hw_h9/random_backpressure_matrix.md` |
| Assertion positive bind | PASS | `reports/hw_h9/assertion_execution_matrix.md` |
| Assertion negative tests | PASS, 23/23 target properties triggered | `reports/hw_h9/assertion_negative_results.md` |
| H9 lint/vlogan | PASS | `reports/hw_h9/lint_results.md` |
| H9 DC structural check | PASS | `reports/hw_h9/synth_check.md` |
| Stage5/6/7/8 regressions | PASS | stage report directories |

## PDK Boundary

No PDK-dependent step was run. DC checks were limited to
analyze/elaborate/link/check_design and hierarchy. `TECH_LIB_ROOT` was not set.
No SRAM macro binding, STA, floorplan, placement, CTS, routing, area, power,
frequency, WNS, TNS, or PPA conclusion is reported.

## Acceptance Decision

The regression bundle passed, but Hardware Stage H9 remains:

```text
HW-H9 IN PROGRESS, NOT ACCEPTED
```

Reason: the strict final-acceptance request still requires independent
multi-head/full-layer reset injection coverage and broad multi-endpoint
multi-head/full-layer random-backpressure coverage. The implemented closure
adds direct H9 datapath reset/random stress and complete assertion bind/negative
evidence; it does not yet fully close those upper-wrapper coverage requirements.
