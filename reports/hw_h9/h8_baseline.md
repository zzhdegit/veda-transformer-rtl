# Hardware Stage H9 H8 Baseline

Source baseline:

- Branch: `stage8-paper-pe-array`
- Commit/tag: `6881529`, `stage8-paper-array-correctness-accepted`
- Report: `reports/stage_08/summary.md`
- Result: Stage 8 PASS

H8 schedule:

```text
QK complete
-> existing Softmax/SFU
-> sV
```

## RTL Counter Baseline

From `reports/stage_08/cycle_utilization_comparison.md`.

| Config | Seq Len | Total | QK | Softmax Reduction/Norm | sV | Paper Active | Inner | Outer | Tail Masked | Mode Switch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D8 single-head | 8 | 693 | 472 | 174 | 94 | 1220 | 1060 | 160 | 4800 | 7 |
| D16 single-head | 8 | 1250 | 936 | 174 | 187 | 2440 | 2120 | 320 | 9600 | 7 |

## Multi-Head H8 RTL Baseline

| Config | Total | Per-Head Attention | Cache Read | Cache Write | PE Stall | SFU Stall | Paper Active | Inner | Outer | Tail Masked |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H1/D8 | 4553 | 4398 | 576 | 64 | 2356 | 300 | 2196 | 1908 | 288 | 8640 |
| H2/D8 | 9095 | 8795 | 1152 | 128 | 4712 | 600 | 4392 | 3816 | 576 | 17280 |
| H4/D8 | 18189 | 17602 | 2304 | 256 | 9424 | 1200 | 8784 | 7632 | 1152 | 34560 |
| H2/D16 | 16851 | 16281 | 2304 | 256 | 9424 | 600 | 8784 | 7632 | 1152 | 34560 |

H8 limitation confirmed: the Stage 8D adapter is correctness-first and maps current `PE_NUM=8` tiles into low paper-array lanes. SFU-PE interleaving is not implemented in H8.
