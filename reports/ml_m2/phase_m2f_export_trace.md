# ML-M2F Report: FP16 Export and Hardware-Aware Traces

## Result

Formal FP16 export and hardware-aware traces completed from the formal best
TinyStories checkpoint.

## Exported Tensor Count

The export path emits 12 tensors:

- token embedding;
- position embedding;
- final RMSNorm gamma;
- LM head;
- norm1 gamma;
- WQ;
- WK;
- WV;
- WO;
- norm2 gamma;
- W1;
- W2.

```text
export_dir=D:/IC_Workspace/VEDA_artifacts/ml_m2/formal/exports/fp16_best
export_manifest_sha256=5cdff9e7332daa2aea0d452478286ad9a43cd1e0bed33493fa2e8770885708c4
```

## Layout

PyTorch `Linear.weight` is validated as `[output_index][input_index]`, matching
the RTL layout `weight[output_index][input_index]`. No transpose is applied for
ML-M2 tensors.

## Hardware-Aware Path

The hardware-aware layer imports the accepted repository bit model:

- `model.transformer.transformer_layer_reference.TransformerLayerReference`
- `model.projection.projection_reference`
- `model.projection.fp32_fp16_reference`
- `model.arithmetic.fp16_fp32_reference`

It simulates token-by-token Stage 7 behavior with FP16 hidden/weights, FP32
accumulation/residuals, FP16 Q/K/V and concat boundaries, ReLU FP16 activation
boundary, and append-only KV cache.

## Trace Coverage

Formal traces cover prompt lengths 1, 2, 8, and 16. Each trace has 40 nodes and
includes:

```text
token_ids
position_ids
token_embedding
position_embedding
layer_input
rmsnorm1_input/output
Q/K/V projection FP32
Q/K/V FP16
score
scaled score
softmax probability
per-head output
concat FP32/FP16
WO
residual1
rmsnorm2
W1
ReLU
activation FP16
W2
residual2
layer_output
final_norm
logits
top-k
next_token
K/V cache final snapshot
K/V cache history after each token
valid_seq_len
```

## Error Summary

```text
max_pytorch_vs_fp16_weight_logit_error=0.0012482404708862305
max_pytorch_vs_hardware_aware_logit_error=0.0021828413009643555
max_layer_output_error=0.0002460405230522156
max_k_cache_error=0.0009813308715820312
max_v_cache_error=0.0004578828811645508
top1_agreement=1.0 for all traced prompt lengths
top5_overlap=1.0 for all traced prompt lengths
first_differing_token=-1 for all traced prompt lengths
```

ML-M2F did not run real RTL co-simulation.
