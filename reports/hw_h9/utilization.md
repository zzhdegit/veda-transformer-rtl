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
single-head interleaved path. Full multi-head/full-layer utilization acceptance
remains open.
