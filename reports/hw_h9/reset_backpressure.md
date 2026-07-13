# Hardware Stage H9 Reset And Backpressure

Status: checkpoint partial.

Model coverage:

- score packet order under bounded FIFO accounting;
- probability packet order under bounded FIFO accounting;
- non-empty sequence tails;
- D_HEAD tails.

RTL checkpoint coverage:

- Score buffer ready/valid stall stability: PASS.
- Probability FIFO ready/valid stall stability: PASS.
- Single-head output-ready smoke path: PASS for D_HEAD=8, 16, and 64.
- Matched single-head deterministic output/done backpressure subset: PASS for
  D_HEAD=8, 16, and 64 at seq16 and seq32.

Matched deterministic subset:

| D_HEAD | Seq | Paper staged RTL | Paper interleaved RTL | H9 output stalls |
|---:|---:|---:|---:|---:|
| 8 | 16 | 1363 | 1170 | 1 |
| 8 | 32 | 2708 | 2209 | 0 |
| 16 | 16 | 2472 | 1172 | 1 |
| 16 | 32 | 4920 | 2213 | 2 |
| 64 | 16 | 9131 | 1187 | 4 |
| 64 | 32 | 18200 | 2225 | 2 |

Open acceptance coverage:

- Reset interrupt matrix across score load, QK, non-empty score FIFO, SFU max,
  exp/sum, normalization, non-empty probability FIFO, sV, output stall, and
  done stall.
- Deterministic producer/SFU/probability-consumer/output/done stall matrix.
- Random backpressure with saved seeds.
- Deadlock timeout coverage across the full H9 required configuration set.
