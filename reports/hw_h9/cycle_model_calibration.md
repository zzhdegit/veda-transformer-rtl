# Hardware Stage H9 Cycle Model Calibration

Status: calibrated to the matched single-head RTL A/B total-cycle interval.

Commands executed:

```text
python3 scripts/sim/run_hw_h9_tests.py
python3 model/attention/paper_interleaved_cycle_model.py
make hw-h9-rtl-sim
```

The Python cycle model uses the matched RTL A/B interval as its default
abstraction. The old structural table is retained only as historical trend
evidence in older reports.

## Calibrated Equations

Let `T = ceil(D_HEAD / 8)` and `S = seq_len`.

```text
paper staged RTL cycles      = (69*T + 15)*S + (5*T + 14) - (12 if S == 1 else 0)
paper interleaved RTL cycles = 65*S + 127 + 2*T
```

The equations include the matched testbench interval for array command
handshake, result latency, scaler/exp/reciprocal latency, staged SRAM replay
latency, WAIT/ACCEPT controller states, tiled output retire, output/done
handshake, D_HEAD tiling, mode switch, fixed startup, and drain overhead.

The model is not hard-coded per table entry; the same formulas generate all
reported D_HEAD and sequence points.

## Calibration Table

| D_HEAD | Seq | RTL staged | Model staged | Delta | RTL H9 interleaved | Model H9 interleaved | Delta | Relative error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 91 | 91 | 0 | 194 | 194 | 0 | 0.00% |
| 8 | 2 | 187 | 187 | 0 | 259 | 259 | 0 | 0.00% |
| 8 | 8 | 691 | 691 | 0 | 649 | 649 | 0 | 0.00% |
| 8 | 16 | 1363 | 1363 | 0 | 1169 | 1169 | 0 | 0.00% |
| 8 | 32 | 2707 | 2707 | 0 | 2209 | 2209 | 0 | 0.00% |
| 8 | 64 | 5395 | 5395 | 0 | 4289 | 4289 | 0 | 0.00% |
| 16 | 1 | 165 | 165 | 0 | 196 | 196 | 0 | 0.00% |
| 16 | 2 | 330 | 330 | 0 | 261 | 261 | 0 | 0.00% |
| 16 | 8 | 1248 | 1248 | 0 | 651 | 651 | 0 | 0.00% |
| 16 | 16 | 2472 | 2472 | 0 | 1171 | 1171 | 0 | 0.00% |
| 16 | 32 | 4920 | 4920 | 0 | 2211 | 2211 | 0 | 0.00% |
| 16 | 64 | 9816 | 9816 | 0 | 4291 | 4291 | 0 | 0.00% |
| 64 | 1 | 609 | 609 | 0 | 208 | 208 | 0 | 0.00% |
| 64 | 2 | 1188 | 1188 | 0 | 273 | 273 | 0 | 0.00% |
| 64 | 8 | 4590 | 4590 | 0 | 663 | 663 | 0 | 0.00% |
| 64 | 16 | 9126 | 9126 | 0 | 1183 | 1183 | 0 | 0.00% |
| 64 | 32 | 18198 | 18198 | 0 | 2223 | 2223 | 0 | 0.00% |
| 64 | 64 | 36342 | 36342 | 0 | 4303 | 4303 | 0 | 0.00% |

## Scope

Total attention cycles are exact for the matched A/B table above. The model's
`full_array_non_interleaved_cycles` and overlap subtotals remain structural
trend estimates and are not used as acceptance evidence.

## Decision

PREVIOUS COMPARISON WAS NOT APPLES-TO-APPLES.

The old structural cycle model is no longer used to decide whether H9 is faster
than H8 staged. Hardware Stage H9 performance acceptance must use matched RTL
A/B data.
