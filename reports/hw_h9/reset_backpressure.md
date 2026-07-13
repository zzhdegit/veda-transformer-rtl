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

Open acceptance coverage:

- Reset interrupt matrix across score load, QK, non-empty score FIFO, SFU max,
  exp/sum, normalization, non-empty probability FIFO, sV, output stall, and
  done stall.
- Deterministic producer/SFU/probability-consumer/output/done stall matrix.
- Random backpressure with saved seeds.
- Deadlock timeout coverage across the full H9 required configuration set.
