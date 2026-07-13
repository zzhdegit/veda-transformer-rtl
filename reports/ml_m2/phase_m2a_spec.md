# ML-M2A Report: Training and Export Specification

## Result

ML-M2A freezes the model-side contract for Model Stage M2:
Hardware-Matched Language Model Training.

## Inputs Read

- `AGENTS.md`
- `PROJECT_STATE.md`
- `HANDOFF.md`
- `README.md`
- `docs/stage_07/spec.md`
- `reports/stage_07/acceptance_audit.md`
- `reports/stage_07/summary.md`
- `rtl/transformer/transformer_layer.sv`
- `rtl/transformer/rmsnorm_engine.sv`
- `rtl/transformer/ffn_engine.sv`
- `rtl/attention/projection_integrated_mha.sv`
- `model/transformer/`
- `model/projection/`
- `model/attention/`

## Frozen Decision

The first formal ML-M2 model is a one-layer, self-trained, hardware-matched
causal LM:

```text
d_model=64
n_head=8
d_head=8
d_ffn=256
context_length=128
vocab_size=2048
norm=RMSNorm
activation=ReLU
bias=false
position=learned absolute, software-side
```

Public checkpoints from Model Stage M1 retain their roles only:

- SmolLM2-135M: future public checkpoint hardware adaptation target;
- Qwen2.5-0.5B: future KV cache eviction software research model;
- OPT-125M: optional compatibility control;
- Llama-2-7B: VEDA paper reference, not current RTL deployment target.

## Hardware Isolation

ML-M2A does not modify RTL, accepted bit-model files, Hardware Stage H8 files,
or hardware stage reports.

## Acceptance

ML-M2A is complete when these files are committed:

- `docs/ml/ml_m2_spec.md`
- `docs/ml/ml_m2_numeric_contract.md`
- `docs/ml/ml_m2_export_contract.md`
- `reports/ml_m2/phase_m2a_spec.md`

