# ML-M2 Quantization and Hardware-Aware Analysis

## Smoke Checkpoint Comparison

Command used a fixed prompt token pair `[1, 5]` from the smoke checkpoint.

```text
PyTorch path: training model, FP32 weights
Hardware-aware path: Stage 7 bit model with FP16 hidden/weights and accepted
FP16/FP32 boundaries
```

Metrics:

```text
max_abs_error=0.0028839111328125
mean_abs_error=0.0006555751897394657
logits_cosine_similarity=1.0
top1_agreement=1.0
top5_overlap=1.0
first_differing_token=-1
```

The smoke checkpoint is not a quality model. These metrics only validate that
the comparison machinery and hardware-aware path are operational.

