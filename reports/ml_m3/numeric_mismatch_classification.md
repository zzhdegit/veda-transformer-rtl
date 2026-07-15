# ML-M3 Numeric Mismatch Classification

| Field | Value |
|---|---|
| Expected | `0x3d4a2572` (0.0493521168828) |
| Actual | `0x3d4a2576` (0.049352131784) |
| ULP distance | 4 |
| Integer bit-pattern delta | 4 |
| 64-dim mismatch count | 54 |
| H8/H9 identical | True |
| NaN/Inf | expected nan=False inf=False; actual nan=False inf=False |
| Subnormal/FTZ | expected subnormal=False; actual subnormal=False |

## Boundary Table

| Boundary | Expected | H8 | H9 | Match | First divergence |
|---|---:|---:|---:|---|---|
| residual1_fp32_edge | `bdc0ae2f` | `bdc0ae2f` | `bdc0ae2f` | True |  |
| norm2_output_fp16_edge | `b925` | `b925` | `b925` | True |  |
| w2_output_fp32_edge | `3e12e074` | `3e12e075` | `3e12e075` | False | yes |
| residual2_input_lhs_fp32_edge | `bdc0ae2f` | `bdc0ae2f` | `bdc0ae2f` | True |  |
| residual2_input_rhs_fp32_edge | `3e12e074` | `3e12e075` | `3e12e075` | False |  |
| residual2_final_fp32_edge | `3d4a2572` | `3d4a2576` | `3d4a2576` | False |  |

## Classification

Root cause class: **C. RTL common arithmetic path bug**.

H8/H9 are identical and vectors, operands, and lane products match the bit model. The first stable divergent boundary is FFN W2 output, and the first arithmetic divergence is inside the W2 reduction-tree add for row 1, tile base 8, pair 3. A standalone stable fp32_add_wrapper replay of the same operands matches the bit model/NumPy, so the full RTL PE reduction path is consuming a different result under its handshake/stream_reg scheduling.
