# ML-M3 RTL Smoke Results

| Length | Stall | H8 staged | H9 interleaved | H8/H9 SHA match | Output tiles | Done | valid_seq_len |
|---:|---|---|---|---|---:|---:|---:|
| 1 | none | PASS | PASS | True | 8 | 1 | 1 |
| 1 | output_done | PASS | PASS | True | 8 | 1 | 1 |
| 2 | none | PASS | PASS | True | 16 | 2 | 2 |
| 2 | output_done | PASS | PASS | True | 16 | 2 | 2 |
| 8 | none | PASS | PASS | True | 64 | 8 | 8 |
| 8 | output_done | PASS | PASS | True | 64 | 8 | 8 |
| 16 | none | PASS | PASS | True | 128 | 16 | 16 |
| 16 | output_done | PASS | PASS | True | 128 | 16 | 16 |
