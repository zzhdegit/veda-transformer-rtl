# Stage 7A Specification Freeze

## Result

Stage 7A freezes the Pre-Norm Transformer layer contract for implementation and
verification. This phase adds repository-owned Stage 7 specifications and a
Python bit-model framework. It does not claim RTL implementation closure.

## Frozen Layer

```text
n1 = RMSNorm(x)
a  = frozen Stage 6 MHA(n1)
r1 = x + a
n2 = RMSNorm(r1)
h1 = W1(n2)
h  = ReLU(h1)
f  = W2(h)
y  = r1 + f
```

The final top remains a single layer and must instantiate exactly one frozen
Stage 6 MHA instance. Multi-layer Transformer, embedding, LM head, tokenizer,
RoPE, bias, dropout, Post-Norm, LayerNorm, GELU, SiLU, and SwiGLU are out of
scope.

## Numeric Freeze

- External hidden input is FP16.
- Layer input and residual buffers are FP32.
- RMSNorm input is FP32, gamma is FP16, output is FP16.
- RMSNorm sum of squares is dimension-order sequential fused MAC.
- RMSNorm mean scale is exact power-of-two FP32.
- `EPS_FP32 = 32'h3727_C5AC`.
- RMSNorm apply order is `(x * inv_rms) * gamma`.
- Stage 6 MHA consumes norm1 FP16 and produces FP32.
- Residual adds are FP32 with the project add wrapper policy.
- FFN weights and activation buffer are FP16.
- FFN GEMV output is FP32 and uses Stage 2/6 reduction order.
- ReLU consumes FP32 and outputs +0 for negative finite values, signed zeros,
  NaN, and Inf.
- ReLU output is quantized to FP16 before W2.
- Final layer output is FP32.

## Model Artifacts

Stage 7A adds:

- `model/transformer/rmsnorm_reference.py`
- `model/transformer/residual_reference.py`
- `model/transformer/relu_reference.py`
- `model/transformer/ffn_reference.py`
- `model/transformer/transformer_layer_reference.py`
- `model/transformer/transformer_layer_cycle_model.py`

The model reuses existing project references for FP16-to-FP32 conversion,
FP32 add, FP32 fused MAC, FP32-to-FP16 conversion, Stage 2/6 GEMV reduction
order, and Stage 6 projection-integrated MHA behavior.

## Test Freeze

Stage 7A model tests cover:

- D_MODEL 8, 16, and 32;
- D_FFN 32, 64, and 128;
- RMSNorm zero, constant, signed, gamma, and non-finite cases;
- gamma address/layout behavior;
- ReLU signed-zero, negative, positive, NaN, and Inf behavior;
- FFN W1/W2 shape and row-major address behavior;
- residual add mapping and +0 exact-zero behavior;
- integrated Stage 6 MHA reuse inside a one-layer reference model;
- multi-token and cache-full Stage 6 semantics through the Stage 7 wrapper.

## Commands

```bash
python scripts/sim/run_stage7a_tests.py
make stage7a-test
```

Later Stage 7 phases must add RTL simulation, lint/vlogan, and DC structural
checks before any Stage 7 PASS claim.

## Deferred

- RMSNorm RTL.
- Residual and layer buffer RTL.
- FFN/ReLU RTL.
- Full `transformer_layer` integration RTL.
- VCS assertions for Stage 7.
- Stage 7 lint/vlogan and DC checks.
- SRAM macro binding, STA, layout, and PPA.
