# HW-H9-N1 Reproduction

## Source Artifact

The reproducer is the read-only ML-M3 artifact set under:

```text
D:/IC_Workspace/VEDA_artifacts/ml_m3
```

The repair only reads vectors, traces, comparisons, manifests, logs, and
waveforms from that tree. No M3 artifact is modified or committed.

## Observed Historical Failure

- Original tag: `hw-h9-sfu-pe-interleaving-thesis-accepted`
- Original commit: `9e0b4c9ba42356ee68e489e99cc5cf64e94f607e`
- Token: 0
- Dimension: 1
- RTL/H8/H9: `32'h3d4a2576`
- Expected hardware-aware bit model: `32'h3d4a2572`
- Mismatch count: 54/64 output dimensions

First stable boundary:

| Boundary | Result |
|---|---|
| `residual1_fp32_edge` | match |
| `norm2_output_fp16_edge` | match |
| `w2_output_fp32_edge` | mismatch |
| `residual2_final_fp32_edge` | mismatch |

First arithmetic difference:

| Field | Value |
|---|---|
| Path | FFN W2 reduction tree add |
| Cycle | 185551 |
| Tile base | 8 |
| Width | 8 |
| Pair | 3 |
| Operand A | `32'h3c81aa0c` |
| Operand B | `32'h39699f40` |
| Old RTL result | `32'h3c837d4b` |
| Expected result | `32'h3c837d4a` |
