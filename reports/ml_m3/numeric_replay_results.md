# ML-M3 Numeric Replay Results

## Operation

First divergent W2 row=1 tile base=8 reduction-tree width=8 pair=3 add.

| Field | Value |
|---|---|
| Operand A | `0x3c81aa0c` (0.0158281549811) |
| Operand B | `0x39699f40` (0.000222799368203) |
| Current bit-model RNE result | `0x3c837d4a` (0.016050953418) |
| NumPy float32 result | `0x3c837d4a` |
| RTL wrapper result | `0x3c837d4a` (documented rnd=4) |
| Direct DW constant-rnd result | `0x3c837d4a` (rnd=4) |
| Wrapper matches current bit model | True |
| Direct rnd0 matches current bit model | True |
| Direct rnd4 matches wrapper | False |
| Direct constant-rnd matches wrapper | True |

## Direct DesignWare Rounding Sweep

| rnd | result | status | invalid |
|---:|---|---|---|
| 0 | `0x3c837d4a` | `0x20` | False |
| 1 | `0x3c837d4a` | `0x20` | False |
| 2 | `0x3c837d4b` | `0x20` | False |
| 3 | `0x3c837d4a` | `0x20` | False |
| 4 | `0x3c837d4b` | `0x20` | False |
| 5 | `0x3c837d4b` | `0x20` | False |
| 6 | `0x3c837d4a` | `0x20` | False |
| 7 | `0x3c837d4a` | `0x20` | False |

## Interpretation

The standalone replay isolates the first divergent arithmetic operation from the full transformer run. The current software bit model, NumPy float32, the direct constant-rnd DesignWare add, and the stable `fp32_add_wrapper` transaction agree on the replay result. The full transformer PE reduction path produced the adjacent result for the captured transaction, so the evidence points at the common RTL reduction/stream handshake context rather than a model-side RNE replay error. The variable-rnd sweep is retained only as a diagnostic signal-level probe and is not treated as the project wrapper contract.
