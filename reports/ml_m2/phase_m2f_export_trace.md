# ML-M2F Report: FP16 Export and Hardware-Aware Traces

## Result

ML-M2F adds FP16 export, manifest validation, hardware-aware Stage 7 bit-model
execution, trace export, and small RTL fixture generation.

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

All tensors are exported as FP16 `.npy` and `.hex` files in the artifact
directory. Large exports are ignored by Git.

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
accumulation/residuals, FP16 Q/K/V and concat boundaries, and append-only KV
cache.

## Trace Coverage

Trace manifests include model-level, layer-level, and cache-level metadata with
shape, dtype, checksum, stage, token index, and layer index.

Recorded smoke artifact run:

```text
exported_tensors=12
trace_node_count=35
trace_valid_seq_len=2
trace_path=build/ml_m2_artifacts/traces/smoke_trace_manifest.json
```

PyTorch vs hardware-aware smoke logits:

```text
max_abs_error=0.0028839111328125
mean_abs_error=0.0006555751897394657
top1_agreement=1.0
top5_overlap=1.0
first_differing_token=-1
```

## Tests

Tests cover:

- Linear export direction;
- FP16 export manifest and SHA256 validation;
- hardware-aware model execution;
- PyTorch vs hardware-aware metric generation;
- trace manifest node coverage;
- small RTL fixture manifest generation.
