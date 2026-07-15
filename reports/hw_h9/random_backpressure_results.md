# Hardware Stage H9 Backpressure Results

Status: deterministic backpressure passes; broad random backpressure remains
open.

Executed deterministic coverage:

- no external backpressure matched A/B for D_HEAD 8, 16, and 64 at seq 1, 2,
  8, 16, 32, and 64: PASS;
- deterministic output/done ready pattern for D_HEAD 8, 16, and 64 at seq16
  and seq32: PASS;
- Stage 5 multi-head output/done deterministic backpressure through H1/D8,
  H2/D8, H4/D8, H2/D16, H1/D64, and H1/D8 MAX_SEQ_LEN=32: PASS;
- Stage 7D full-layer output/done deterministic backpressure inherited from
  the Stage 7D testbench: PASS.

Representative deterministic matched A/B results:

| D_HEAD | Seq | Staged | Interleaved | Result |
|---:|---:|---:|---:|---|
| 8 | 16 | 1363 | 1170 | PASS |
| 8 | 32 | 2708 | 2209 | PASS |
| 16 | 16 | 2472 | 1172 | PASS |
| 16 | 32 | 4920 | 2213 | PASS |
| 64 | 16 | 9131 | 1187 | PASS |
| 64 | 32 | 18200 | 2225 | PASS |

Still open for acceptance:

- at least 20 fixed random seeds;
- random score/SFU/probability/array/head/final/done ready patterns;
- saved failed seed and transaction trace on failure;
- watchdog derived from maximum legal latency.

Random seed list for this closure:

```text
none
```

No random seed set is accepted for Hardware Stage H9 Final Closure.
