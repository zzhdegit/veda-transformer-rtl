# Hardware Stage H9 Cycle Comparison

Structural model command:

```text
python model/attention/paper_interleaved_cycle_model.py
```

| Seq Len | H8 staged cycles | Full-array non-interleaved | H9 interleaved | QK-SFU overlap | SFU-sV overlap |
|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 32 | 32 | 0 | 0 |
| 2 | 34 | 61 | 50 | 7 | 2 |
| 8 | 118 | 235 | 158 | 49 | 14 |
| 16 | 230 | 467 | 302 | 105 | 30 |
| 32 | 454 | 931 | 590 | 217 | 62 |

Interpretation:

- H9 interleaving reduces cycles relative to the full-array non-interleaved H9
  structural schedule.
- H9 interleaving is not lower than the H8 staged baseline in this model.
- This old structural comparison is not apples-to-apples with RTL and no
  longer controls the HW-H9 performance decision.
- Seq 1 has no overlap and carries the full interleaved setup cost.

The table is structural schedule evidence only. It is not a timing, frequency,
area, power, or PPA result.

## Matched RTL Revision

The closure audit added a matched RTL A/B using the same `single_head_attention`
top, same `ATTENTION_PE_ARCH=PAPER_ARRAY`, same inputs, same DesignWare latency,
same clock/reset, and same output/done ready environment.

PREVIOUS COMPARISON WAS NOT APPLES-TO-APPLES.

Matched RTL totals:

| D_HEAD | Seq | Paper staged RTL | Paper interleaved RTL | Improvement |
|---:|---:|---:|---:|---:|
| 8 | 16 | 1363 | 1169 | 14.2% |
| 8 | 32 | 2707 | 2209 | 18.4% |
| 16 | 16 | 2472 | 1171 | 52.6% |
| 16 | 32 | 4920 | 2211 | 55.1% |
| 64 | 16 | 9126 | 1183 | 87.0% |
| 64 | 32 | 18198 | 2223 | 87.8% |

See `reports/hw_h9/matched_rtl_ab_baseline.md` for the full seq1/2/8/16/32/64
table and deterministic backpressure subset.
