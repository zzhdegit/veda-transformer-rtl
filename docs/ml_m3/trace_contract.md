# ML-M3 Trace Contract

## Vector Cases

Mandatory real-input vector lengths:

```text
1, 2, 8, 16
```

Extended non-blocking vector length:

```text
32
```

Vectors are reconstructed from the frozen Q2 benchmark prompt rule and saved
under:

```text
D:/IC_Workspace/VEDA_artifacts/ml_m3/vectors
```

Each vector uses Stage7D-style records:

```text
C n_head d_head d_model d_ffn max_seq_len
W kind output_index input_index fp16_bits
T token_index meta
H dim hidden_fp16_bits
O dim expected_layer_output_fp32_bits
```

## Trace and Capture Artifacts

The M3 RTL testbench captures final RTL layer output as:

```text
R token_index dim fp32_bits
```

Captures are written to:

```text
D:/IC_Workspace/VEDA_artifacts/ml_m3/traces
```

Because one-token smoke currently fails at token 0 dimension 1, captures are
partial and must not be used as complete RTL-assisted logits.

## Required Comparison Rules

- PyTorch FP32 vs FP16-weight PyTorch: report error statistics.
- PyTorch/FP16 vs hardware-aware bit model: report error statistics.
- Hardware-aware bit model vs H8/H9 RTL: bit-exact required.
- H8 RTL vs H9 RTL: bit-exact required.

Current status: H8/H9 captured prefix is identical, but both fail bit-exact
comparison against the hardware-aware bit model.
