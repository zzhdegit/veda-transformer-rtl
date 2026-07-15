# Hardware Stage H9 Utilization

Structural model for D_HEAD=64:

| Seq Len | Array Active | Array Utilization | SFU Active | SFU Utilization | Score FIFO Peak | Probability FIFO Peak |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 13 | 0.406 | 16 | 0.500 | 1 | 1 |
| 2 | 26 | 0.520 | 30 | 0.600 | 1 | 1 |
| 8 | 104 | 0.658 | 114 | 0.722 | 1 | 1 |
| 16 | 208 | 0.689 | 226 | 0.748 | 1 | 1 |
| 32 | 416 | 0.705 | 450 | 0.763 | 1 | 1 |

Current H8 RTL utilization is recorded in `reports/hw_h9/h8_baseline.md`.

RTL smoke evidence:

```text
tb_h9_single_head D_HEAD=8  group0=408 group1=408 total=649
tb_h9_single_head D_HEAD=16 group0=408 group1=408 total=651
tb_h9_single_head D_HEAD=64 group0=408 group1=408 total=663
```

The RTL smoke counters prove both groups are active in the checkpoint
single-head interleaved path. Final-closure multi-head and full-layer RTL runs
also pass, but detailed multi-head/full-layer utilization attribution remains
informational rather than an H9 PASS condition because reset/random/assertion
coverage is still incomplete.

## Matched RTL Closure Update

The matched RTL A/B run reports both groups active for every interleaved
single-head configuration. Selected no-backpressure points:

| D_HEAD | Seq | Total | Array active | Array idle | Group0 active | Group1 active | QK-SFU overlap | SFU-sV overlap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 1169 | 1082 | 619 | 744 | 744 | 239 | 130 |
| 8 | 32 | 2209 | 2058 | 1195 | 1416 | 1416 | 447 | 258 |
| 16 | 16 | 1171 | 1082 | 1149 | 744 | 744 | 239 | 130 |
| 16 | 32 | 2211 | 2058 | 2237 | 1416 | 1416 | 447 | 258 |
| 64 | 16 | 1183 | 1082 | 4329 | 744 | 744 | 239 | 130 |
| 64 | 32 | 2223 | 2058 | 8489 | 1416 | 1416 | 447 | 258 |

The large D_HEAD=64 idle count includes testbench load/start-to-output interval
accounting and tiled output retirement around the single-head top; it is not a
PPA utilization claim.
