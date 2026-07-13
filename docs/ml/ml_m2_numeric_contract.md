# ML-M2 Numeric Contract

## Training Path

Training may use FP32, BF16, or mixed precision. The default CPU smoke path is
FP32. GPU training should prefer BF16 only when the hardware supports it.

The first ML-M2 implementation does not use FP16 training by default and does
not introduce quantization-aware training.

## Inference Path

Software inference runs the same architecture with dropout disabled. It supports:

- full-sequence causal forward;
- incremental single-token decode with an append-only per-layer KV cache;
- greedy generation for deterministic prompts.

## Export Path

Export converts trainable tensors to FP16 for RTL-facing artifacts:

- hidden input to the RTL layer is FP16;
- RMSNorm gamma is FP16;
- WQ/WK/WV/WO are FP16;
- W1/W2 are FP16;
- Q/K/V projection results are quantized from FP32 to FP16;
- K/V cache stores FP16;
- attention scores and softmax path are FP32 in the hardware-aware model;
- head concat is quantized from FP32 to FP16 before W_O;
- residuals and final layer output are FP32.

## Rounding and Bit-Model Boundary

The hardware-aware path must import and reuse the accepted repository bit-model
helpers under `model/` for FP32/FP16 conversion and Stage 7 layer composition.
It must not silently replace the accepted RTL numeric policy with an unrelated
floating-point approximation.

## RMSNorm

Default epsilon:

```text
EPS_FP32 = float32(1.0e-5)
```

RMSNorm apply order is:

```text
(x * inv_rms) * gamma
```

This matches Stage 7. The PyTorch training model may use native tensor
arithmetic for optimization, while export and hardware-aware trace generation
use the repository bit-model policy.

## ReLU

Training model ReLU is standard finite `max(x, 0)`. The hardware-aware path
uses the Stage 7 bit-model ReLU/quantization behavior, including the accepted
NaN/Inf invalid handling.

## Validation Metrics

ML-M2 reports:

- train loss;
- validation loss;
- perplexity;
- FP32 vs FP16-weight error;
- PyTorch vs hardware-aware error;
- first differing token;
- top-1 agreement;
- top-5 overlap;
- logits max absolute error;
- logits cosine similarity;
- layer output error;
- KV cache error.

