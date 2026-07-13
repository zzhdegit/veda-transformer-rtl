# Stage 8B Paper Array Model Report

## Result

Stage 8B paper-array Python/model regression: PASS

Date: 2026-07-13

Branch: `stage8-paper-pe-array`

## Added Model Files

- `model/pe_array/paper_pe_reference.py`
- `model/pe_array/paper_pe_group_reference.py`
- `model/pe_array/paper_array_8x8x2_reference.py`
- `model/pe_array/paper_array_mapping.py`
- `model/pe_array/paper_array_cycle_model.py`
- `model/pe_array/paper_array_compare_legacy.py`

The model explicitly constructs:

- 8 rows.
- 8 columns.
- 2 PE groups.
- 128 PE cell objects.
- Type-A and Type-B identities.
- Row-level L1 reduction.
- Group-level L2 reduction.
- Group combine stage.
- Inner-product and outer-product modes.
- Group, row, and column masks.
- Tail tiles and multi-tile vectors.
- Mode switch tracking.
- Reset behavior.

## Test Coverage

Test file: `tb/model/test_stage8_paper_array.py`

Covered cases:

- all zero;
- identity;
- powers of two;
- positive/negative mixed values;
- cancellation;
- deterministic dense values;
- signed zero;
- FP16 min normal;
- FP16 max normal;
- multiple tiles;
- partial rows;
- partial columns;
- only group 0 active;
- only group 1 active;
- both groups active;
- mode switch;
- repeated command;
- reset.

The model does not use `numpy.matmul` or `torch.matmul`.

## Regression Commands

Host:

- `python scripts/sim/run_stage8b_tests.py`: PASS

Host Makefile:

- `make stage8b-test`: unavailable because host PowerShell environment does not provide `make`.

Docker:

- `docker exec nailong bash -lc 'cd /workspace/VEDA && make stage8b-test'`: PASS

Docker ran:

- `python3 scripts/sim/run_stage8a_tests.py`
- `python3 scripts/sim/run_stage8b_tests.py`
- `python3 -m py_compile` on Stage 8B model files
- `python3 model/pe_array/paper_array_cycle_model.py`
- `python3 model/pe_array/paper_array_compare_legacy.py`

## Legacy Comparison Snapshot

The current deterministic comparison case reports bit-exact matches between
the Stage 8 paper-array model and the legacy PE model for both inner and outer
paths:

- inner bit_exact: True
- outer bit_exact: True
- max_abs_error: 0.0
- MAE: 0.0
- RMSE: 0.0
- relative_l2: 0.0
- max ULP: 0
- argmax match: True

This snapshot is not a claim that every future Stage 8 mapping will preserve
legacy add order. If RTL mapping uses a different order, the comparison report
must record the nonzero metrics instead of applying tolerance to RTL.

## Limitations

- This is a Python bit model only; no Stage 8 RTL has been implemented in this phase.
- Cycle counts are structural no-stall estimates, not measured RTL counters.
- Status bit encoding remains a repository integration decision for the RTL stage.
- SFU-PE interleaving remains out of scope.
- KV cache eviction remains out of scope.
