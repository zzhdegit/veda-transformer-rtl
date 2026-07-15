# ML-M3 First Divergence Trace

The first stable boundary divergence is `w2_output_fp32_edge`; earlier `residual1_fp32_edge` and `norm2_output_fp16_edge` match.

## First Arithmetic Divergence

| Field | Value |
|---|---|
| Node | W2 reduction tree add |
| Cycle | 185551 |
| Tile base | 8 |
| Width/pair | 8 / 3 |
| Operand A | `0x3c81aa0c` |
| Operand B | `0x39699f40` |
| RTL result | `0x3c837d4b` |
| Bit-model result | `0x3c837d4a` |
| ULP distance | 1 |
