# ML-M3 Hybrid Next-Token Results

| Case | Prompt token IDs | Bit top-1 | H8 top-1 | H9 top-1 | Top-1 pass |
|---|---|---:|---:|---:|---|
| len_1 | [1] | 230 | 230 | 230 | True |
| len_8 | [1, 230, 5, 240, 5, 54, 5, 204] | 5 | 5 | 5 | True |
| len_16 | [1, 230, 5, 240, 5, 54, 5, 204, 5, 206, 5, 107, 5, 54, 5, 556] | 5 | 5 | 5 | True |

## Continuous Two-Step

| Step | Prompt token IDs | Bit top-1 | H8 top-1 | H9 top-1 | Result |
|---:|---|---:|---:|---:|---|
| 0 | [1] | 230 | 230 | 230 | PASS |
| 1 | [1, 230] | 5 | 5 | 5 | PASS |
