# HW-H9-N1 Performance Regression

The repair changes only the `DW_fp_add.rnd` encoding and simulation-only
assertions. It does not change the ready/valid protocol, wrapper latency,
reduction FSM, external transaction count, or H9 schedule.

Matched H9 no-stall RTL A/B totals remain:

| D_HEAD | seq16 staged | seq32 staged | seq16 interleaved | seq32 interleaved |
|---:|---:|---:|---:|---:|
| 8 | 1363 | 2707 | 1169 | 2209 |
| 16 | 2472 | 4920 | 1171 | 2211 |
| 64 | 9126 | 18198 | 1183 | 2223 |

The H9 cycle model calibration remains total-cycle delta 0 in the thesis
acceptance run.

Backpressure variants intentionally show output-stall deltas in some rows; they
are not the no-stall matched performance contract.
