# ML-M3 Incremental KV Results

Software full-vs-incremental reference: **PASS** (valid_seq_len=16, max_abs=3.618001937866211e-05).

| Length | Stall | Schedule | Token lines | Output lanes | Output tiles | Done count | valid_seq_len | Result |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | none | staged | 1 | 64 | 8 | 1 | 1 | PASS |
| 1 | none | interleaved | 1 | 64 | 8 | 1 | 1 | PASS |
| 1 | output_done | staged | 1 | 64 | 8 | 1 | 1 | PASS |
| 1 | output_done | interleaved | 1 | 64 | 8 | 1 | 1 | PASS |
| 2 | none | staged | 2 | 128 | 16 | 2 | 2 | PASS |
| 2 | none | interleaved | 2 | 128 | 16 | 2 | 2 | PASS |
| 2 | output_done | staged | 2 | 128 | 16 | 2 | 2 | PASS |
| 2 | output_done | interleaved | 2 | 128 | 16 | 2 | 2 | PASS |
| 8 | none | staged | 8 | 512 | 64 | 8 | 8 | PASS |
| 8 | none | interleaved | 8 | 512 | 64 | 8 | 8 | PASS |
| 8 | output_done | staged | 8 | 512 | 64 | 8 | 8 | PASS |
| 8 | output_done | interleaved | 8 | 512 | 64 | 8 | 8 | PASS |
| 16 | none | staged | 16 | 1024 | 128 | 16 | 16 | PASS |
| 16 | none | interleaved | 16 | 1024 | 128 | 16 | 16 | PASS |
| 16 | output_done | staged | 16 | 1024 | 128 | 16 | 16 | PASS |
| 16 | output_done | interleaved | 16 | 1024 | 128 | 16 | 16 | PASS |
