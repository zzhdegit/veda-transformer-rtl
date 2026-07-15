# ML-M3 Node Comparison

The production RTL was not modified. Directly observed real-RTL boundaries were
captured through read-only hierarchical references in the model-line M3
testbench.

| Schedule | Length | Result | Boundary line |
|---|---:|---|---|
| staged | 1 | DIAGNOSTIC | `ML_M3_BOUNDARY_OBS norm1=1 mha=1 residual1=1 norm2=1 ffn=1 residual2=1 output_tiles=8 done=1` |
| interleaved | 1 | DIAGNOSTIC | `ML_M3_BOUNDARY_OBS norm1=1 mha=1 residual1=1 norm2=1 ffn=1 residual2=1 output_tiles=8 done=1` |

First stable boundary divergence:

| Boundary | Expected | H8 | H9 | Match |
|---|---:|---:|---:|---|
| `residual1_fp32_edge` | `bdc0ae2f` | `bdc0ae2f` | `bdc0ae2f` | True |
| `norm2_output_fp16_edge` | `b925` | `b925` | `b925` | True |
| `w2_output_fp32_edge` | `3e12e074` | `3e12e075` | `3e12e075` | False |
| `residual2_final_fp32_edge` | `3d4a2572` | `3d4a2576` | `3d4a2576` | False |

The first divergent arithmetic operation is documented in
`reports/ml_m3/first_divergence_trace.md`.
