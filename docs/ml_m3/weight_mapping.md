# ML-M3 Weight Mapping

The Q2 model exports 12 FP16 tensors. Only 8 tensors enter the RTL
Transformer-layer boundary; embedding, learned position, final RMSNorm, and
tied LM head remain software-side.

RTL weight layout is:

```text
weight[output_index][input_index]
```

No transpose is applied for PyTorch `Linear.weight`.

| Tensor | Kind | Shape | Source state_dict |
|---|---:|---:|---|
| WQ | 0 | [64, 64] | `layers.0.attn.wq.weight` |
| WK | 1 | [64, 64] | `layers.0.attn.wk.weight` |
| WV | 2 | [64, 64] | `layers.0.attn.wv.weight` |
| WO | 3 | [64, 64] | `layers.0.attn.wo.weight` |
| NORM1_GAMMA | 4 | [64] | `layers.0.norm1.weight` |
| NORM2_GAMMA | 5 | [64] | `layers.0.norm2.weight` |
| FFN_W1 | 6 | [256, 64] | `layers.0.ffn.w1.weight` |
| FFN_W2 | 7 | [64, 256] | `layers.0.ffn.w2.weight` |

Audit result: PASS. See `reports/ml_m3/weight_mapping_audit.md` and
`D:/IC_Workspace/VEDA_artifacts/ml_m3/manifests/artifact_audit.json`.
