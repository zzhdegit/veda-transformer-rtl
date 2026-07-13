# Hardware Stage H9 Cycle Model Calibration

Status: partially calibrated by measurement; model code remains structural.

Commands:

```text
python model/attention/paper_interleaved_cycle_model.py
docker exec -w /workspace/VEDA nailong make hw-h9-rtl-sim
```

The current Python model is a structural schedule model. It is useful for
checking overlap ordering, but it undercounts RTL cycles because it omits or
abstracts:

- paper array command/result handshakes;
- DesignWare valid latency in scaler, exp, reciprocal, and arithmetic wrappers;
- staged score/probability SRAM read latency;
- controller read/wait/accept states;
- tiled output retirement;
- RTL output/done backpressure cycles;
- D_HEAD-dependent staged low-lane tiling.

## D_HEAD=8

| Seq | Model staged | RTL staged | Delta | Model H9 | RTL H9 | Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 91 | 71 | 32 | 194 | 162 |
| 2 | 34 | 187 | 153 | 50 | 259 | 209 |
| 8 | 118 | 691 | 573 | 158 | 649 | 491 |
| 16 | 230 | 1363 | 1133 | 302 | 1169 | 867 |
| 32 | 454 | 2707 | 2253 | 590 | 2209 | 1619 |

## D_HEAD=16

| Seq | Model staged | RTL staged | Delta | Model H9 | RTL H9 | Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 165 | 145 | 32 | 196 | 164 |
| 2 | 34 | 330 | 296 | 50 | 261 | 211 |
| 8 | 118 | 1248 | 1130 | 158 | 651 | 493 |
| 16 | 230 | 2472 | 2242 | 302 | 1171 | 869 |
| 32 | 454 | 4920 | 4466 | 590 | 2211 | 1621 |

## D_HEAD=64

| Seq | Model staged | RTL staged | Delta | Model H9 | RTL H9 | Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 609 | 589 | 32 | 208 | 176 |
| 2 | 34 | 1188 | 1154 | 50 | 273 | 223 |
| 8 | 118 | 4590 | 4472 | 158 | 663 | 505 |
| 16 | 230 | 9126 | 8896 | 302 | 1183 | 881 |
| 32 | 454 | 18198 | 17744 | 590 | 2223 | 1633 |

## Calibration Decision

PREVIOUS COMPARISON WAS NOT APPLES-TO-APPLES.

The structural model predicted H9 as slower than the H8 staged baseline, but
the matched RTL A/B shows H9 interleaved is faster for the required seq16 and
seq32 cases. The Python model will remain labeled structural until a later H9
closure commit updates the model equations to reproduce the RTL counter
intervals exactly.

Current performance acceptance evidence must therefore use
`reports/hw_h9/matched_rtl_ab_baseline.md`, not the old structural table alone.
