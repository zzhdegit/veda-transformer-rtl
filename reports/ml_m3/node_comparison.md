# ML-M3 Node Comparison

The production RTL was not modified. Directly observed real-RTL boundaries are top output, top done, and internal ready/done handshakes exposed through read-only hierarchical references in the M3 testbench.

| Schedule | Length | Stall | Result | Boundary line |
|---|---:|---|---|---|
| staged | 32 | none | PASS | `ML_M3_BOUNDARY_OBS norm1=32 mha=32 residual1=32 norm2=32 ffn=32 residual2=32 output_tiles=256 done=32` |
| interleaved | 32 | none | PASS | `ML_M3_BOUNDARY_OBS norm1=32 mha=32 residual1=32 norm2=32 ffn=32 residual2=32 output_tiles=256 done=32` |
