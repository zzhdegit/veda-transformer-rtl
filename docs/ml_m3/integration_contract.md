# ML-M3 Integration Contract

## Reference Chain

Reference 0: PyTorch FP32

- Loads the frozen Q2 checkpoint.
- Used for quality and floating-point error context.
- Not required to be bit-exact with RTL.

Reference 1: FP16-weight PyTorch

- Rounds model weights to FP16 and keeps PyTorch operator order.
- Separates exported-weight quantization error from bit-model effects.

Reference 2: Hardware-aware bit model

- Reuses the existing accepted `model/transformer`, `model/projection`, and
  `model/attention` reference path through `ml/cosim/hardware_aware_layer.py`.
- Uses FP16 hidden/weights and FP32 accumulation/residual boundaries.
- Golden reference for RTL bit-exact comparison.

Reference 3: H8 staged RTL

- `transformer_layer.sv`
- `ATTENTION_PE_ARCH=PAPER_ARRAY`
- `ATTENTION_SCHEDULE=STAGED`

Reference 4: H9 interleaved RTL

- `transformer_layer.sv`
- `ATTENTION_PE_ARCH=PAPER_ARRAY`
- `ATTENTION_SCHEDULE=INTERLEAVED`

## File-System Boundary

VCS compiles read hardware RTL from `/workspace/VEDA`, while testbench,
vectors, logs, captures, and temporary build output live outside the hardware
repository:

```text
testbench=/workspace/VEDA_ml_m2/ml/cosim/rtl_tb/tb_ml_m3_transformer_layer.sv
vectors=/workspace/VEDA_artifacts/ml_m3/vectors
logs=/workspace/VEDA_artifacts/ml_m3/rtl_logs
captures=/workspace/VEDA_artifacts/ml_m3/traces
build=/workspace/VEDA_artifacts/ml_m3/temporary_build
```

## Repaired Baseline Result

Both H8 staged and H9 interleaved elaborate and pass for `N_HEAD=8`,
`D_HEAD=8`, `D_MODEL=64`, `D_FFN=256`, `MAX_SEQ_LEN=128` using the repaired
hardware tag `hw-h9-real-weight-numeric-repair-accepted`.

```text
lengths=1,2,8,16
stall_modes=none,output_done
bit_model == H8 RTL == H9 RTL
```

Length32 no-stall also passes as an extended co-simulation case.
