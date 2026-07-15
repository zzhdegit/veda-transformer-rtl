# ML-M3 Numeric Contract

ML-M3 keeps the Stage 7/H9 arithmetic contract for RTL comparison:

- hardware-aware bit model, H8 staged RTL, and H9 interleaved RTL must be bit-exact at the Transformer layer output;
- PyTorch FP32 and exported-FP16 PyTorch are diagnostic references only and are compared with error metrics;
- no ULP tolerance is accepted for RTL-vs-bit-model closure;
- Q2 checkpoint, tokenizer, vectors, and FP16 weights are frozen.

The current numeric-alignment investigation did not change this contract. The reproduced one-token mismatch remains a failure:

- token `0`, output dimension `1`;
- bit-model expected `0x3d4a2572`;
- H8/H9 RTL actual `0x3d4a2576`;
- final output distance `4` ULP;
- full 64-dim diagnostic capture contains `54` mismatched dimensions.

The first stable divergent boundary is the FFN W2 output. Residual1, RMSNorm2 input/output, W2 FP16 operands, and W2 lane products match the hardware-aware bit model. The first divergent arithmetic operation is in the common W2 reduction-tree add path, not in H9 interleaving.

Current status:

`MODEL STAGE M3 IN PROGRESS - HARDWARE NUMERIC FIX REQUIRED`

Model-side code must not mask this by changing expected vectors, using RTL outputs as golden data, or allowing ULP tolerance.
