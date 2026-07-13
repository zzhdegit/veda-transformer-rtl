# Stage 7 Specification: Pre-Norm Transformer Layer

## Scope

Stage 7 implements one decoder-style Pre-Norm Transformer layer around the
accepted Stage 6 projection-integrated MHA top.

Frozen mathematical structure:

```text
n1 = RMSNorm(x)
a  = MHA(n1)
r1 = x + a
n2 = RMSNorm(r1)
h1 = W1(n2)
h  = ReLU(h1)
f  = W2(h)
y  = r1 + f
```

Stage 7 does not implement LayerNorm, Post-Norm, GELU, SiLU, SwiGLU, bias,
dropout, RoPE, embedding, LM head, tokenizer, multiple layers, SRAM macro
binding, STA, P&R, formal PPA, area, power, frequency, or WNS closure.

## Parameters

Frozen relations:

```text
D_MODEL = N_HEAD * D_HEAD
D_FFN   = 4 * D_MODEL
```

`D_MODEL` must be a power of two for the first Stage 7 implementation. Checked
configurations:

```text
H1/D8   D_MODEL=8   D_FFN=32
H2/D8   D_MODEL=16  D_FFN=64
H4/D8   D_MODEL=32  D_FFN=128
H2/D16  D_MODEL=32  D_FFN=128
```

The primary dense deterministic end-to-end configuration is H2/D8.

## Numeric Boundaries

Layer input is external FP16 hidden state. It is exactly expanded with the
existing `fp16_to_fp32` policy and stored as:

```text
x_fp32[D_MODEL]
```

RMSNorm input is FP32. Gamma is FP16 and is exactly expanded to FP32 before
multiplication. RMSNorm output to MHA/FFN is quantized to FP16 with the existing
RNE/FTZ/saturation FP32-to-FP16 policy.

MHA is the frozen Stage 6 `projection_integrated_mha` behavior:

```text
norm1_fp16 -> Stage 6 MHA -> mha_output_fp32
```

Residual outputs remain FP32 and are not quantized after residual add.

FFN inputs are FP16, weights are FP16, and GEMV accumulates in FP32 using the
Stage 2/6 tile reduction order. ReLU consumes FP32 and is quantized to FP16
before W2. Final layer output remains tiled FP32.

## RMSNorm

Default epsilon is the FP32 bit pattern:

```text
EPS_FP32 = 32'h3727_C5AC  // float32(1.0e-5)
```

For input vector `x_fp32`:

```text
acc_0 = +0
acc_(i+1) = FMA(x[i], x[i], acc_i)
sum_sq = acc_D_MODEL
mean_sq = sum_sq * 2^(-log2(D_MODEL))
den = sqrt(mean_sq + EPS)
inv_rms = 1 / den
norm_fp32[i] = (x_fp32[i] * inv_rms) * gamma_fp32[i]
norm_fp16[i] = fp32_to_fp16(norm_fp32[i])
```

The multiplication order is frozen:

1. `x * inv_rms`
2. previous result `* gamma`

The sum of squares reduction is dimension-order sequential fused MAC. Python
`sum` or library vector reductions are not a golden model for this path.

The power-of-two mean scale must be an exact FP32 power-of-two constant. The
minimum supported constants are 1/8, 1/16, 1/32, 1/64, and 1/128.

## Residual

Residual 1:

```text
residual1_fp32[i] = FP32_ADD(x_fp32[i], mha_output_fp32[i])
```

Residual 2:

```text
layer_output_fp32[i] = FP32_ADD(residual1_fp32[i], ffn2_fp32[i])
```

Both use the project `fp32_add_wrapper` policy. Exact zero residual results are
`+0`.

## FFN

The first Stage 7 FFN is two-layer ReLU without bias:

```text
ffn1_fp32[D_FFN] = W1[D_FFN][D_MODEL] * norm2_fp16[D_MODEL]
relu_fp32[i] = ReLU(ffn1_fp32[i])
activation_fp16[i] = fp32_to_fp16(relu_fp32[i])
ffn2_fp32[D_MODEL] = W2[D_MODEL][D_FFN] * activation_fp16[D_FFN]
```

Weights are output-row-major:

```text
W1[output_index][input_index], output_index < D_FFN,  input_index < D_MODEL
W2[output_index][input_index], output_index < D_MODEL, input_index < D_FFN
```

W1 and W2 must share one Stage 7 FFN GEMV datapath in RTL. The first
implementation may be fully serial.

## ReLU

For finite FP32:

- positive finite: pass through unchanged;
- negative finite: output `+0`;
- `+0` or `-0`: output `+0`.

NaN or Inf is invalid and outputs `+0`.

## Weight Interface

The Stage 7 top uses one external weight-load interface:

- `weight_valid`, `weight_ready`
- `weight_kind`
- `weight_output_index`, `weight_input_index`
- `weight_data_fp16`
- `weight_last`, `weight_commit`

Required logical weight kinds:

- `WQ`, `WK`, `WV`, `WO`
- `NORM1_GAMMA`
- `NORM2_GAMMA`
- `FFN_W1`
- `FFN_W2`

WQ/WK/WV/WO route into the frozen Stage 6 instance. Gamma uses
`output_index` as dimension and requires `input_index == 0`. All weight kinds
maintain valid/complete state. Active token transactions block weight writes.

## Stage 6 Integration

Stage 7 instantiates Stage 6 as a frozen child:

```text
transformer_layer
`-- projection_integrated_mha
```

Stage 7 must not directly rewire Stage 6 internal controllers, PEs, or buffers.
Stage 6 ready/valid semantics, weight layout, QKV/concat numeric boundaries,
current-token causal semantics, all-head atomic commit, and no-next-token-before
done rule remain frozen.

Stage 6 `done_valid` is internal MHA done, not Stage 7 layer done. Stage 7
layer done is emitted only after residual2 output completes.

## Transaction Semantics

Successful token order:

```text
input load
-> RMSNorm1
-> Stage 6 MHA
-> residual1
-> RMSNorm2
-> FFN1
-> ReLU and activation quantization
-> FFN2
-> residual2
-> final FP32 output
-> layer done
```

Stage 6 pre-commit error prevents residual/FFN execution and reports layer
invalid. Post-commit Stage 7 errors do not roll back Stage 6 K/V. Reset clears
top transaction valid state; data memory contents need not be cleared unless a
local module requires it for correctness.

## Counters

Stage 7 final top must expose at least:

- `perf_generation_steps`
- `perf_total_layer_cycles`
- `perf_input_load_cycles`
- `perf_norm1_reduce_cycles`
- `perf_norm1_apply_cycles`
- `perf_mha_cycles`
- `perf_residual1_cycles`
- `perf_norm2_reduce_cycles`
- `perf_norm2_apply_cycles`
- `perf_ffn1_cycles`
- `perf_relu_cycles`
- `perf_activation_quantization_cycles`
- `perf_ffn2_cycles`
- `perf_residual2_cycles`
- `perf_final_output_cycles`
- `perf_norm_stall_cycles`
- `perf_mha_stall_cycles`
- `perf_ffn_pe_stall_cycles`
- `perf_weight_stall_cycles`
- `perf_buffer_stall_cycles`
- `perf_output_stall_cycles`
- `perf_peak_valid_seq_len`

Counters are cumulative from reset unless a test explicitly records deltas.

## Verification Requirements

Stage 7A freezes the bit model and interface contract. Later RTL phases must
add VCS assertion, vlogan lint, and DC analyze/elaborate/link/check_design
coverage before being accepted.

The Stage 7 acceptance baseline is recorded in
`reports/stage_07/acceptance_audit.md`. Future changes that alter the frozen
numeric order, FP16/FP32 boundaries, Stage 6 commit semantics, external
ready/valid behavior, weight layout, or layer done semantics must update this
spec and the bit model, then rerun the full Stage 5/6/7 regression set.

Required model trace nodes:

- `input_fp32`
- `norm1_sum_sq`, `norm1_inv_rms`, `norm1_fp32`, `norm1_fp16`
- `mha_fp32`
- `residual1_fp32`
- `norm2_sum_sq`, `norm2_inv_rms`, `norm2_fp32`, `norm2_fp16`
- `ffn1_fp32`
- `relu_fp32`
- `activation_fp16`
- `ffn2_fp32`
- `final_fp32`

High-precision references may be used for diagnostics only and must not become
RTL tolerance.
