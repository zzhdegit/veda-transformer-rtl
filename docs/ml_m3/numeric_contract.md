# ML-M3 Numeric Contract

ML-M3 keeps the Stage 7/H9 arithmetic contract for RTL comparison:

- hardware-aware bit model, H8 staged RTL, and H9 interleaved RTL must be bit-exact at the Transformer layer output;
- PyTorch FP32 and exported-FP16 PyTorch are diagnostic references only and are compared with error metrics;
- no ULP tolerance is accepted for RTL-vs-bit-model closure;
- Q2 checkpoint, tokenizer, vectors, and FP16 weights are frozen.

The numeric contract is now closed on the repaired H9 baseline
`hw-h9-real-weight-numeric-repair-accepted` at commit
`a54e608a8dc7e63c7e5dd342f8b893bb1e0b7485`.

Current status:

`MODEL STAGE M3 PASS`

Model-side code must not mask future failures by changing expected vectors,
using RTL outputs as golden data, or allowing ULP tolerance. The accepted runs
use bit-exact comparisons for H8 staged RTL, H9 interleaved RTL, and the
hardware-aware bit model.
