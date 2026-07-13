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
- H9 interleaving is not yet lower than the H8 staged baseline in this model.
  This blocks final Hardware Stage H9 acceptance.
- Seq 1 has no overlap and carries the full interleaved setup cost.

The table is structural schedule evidence only. It is not a timing, frequency,
area, power, or PPA result.
