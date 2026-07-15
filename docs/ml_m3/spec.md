# Model Stage M3 Specification

Short id: ML-M3

Name: Real-Weight RTL Co-Simulation and Deployment Validation

Status: IN PROGRESS - RTL/bit-model numeric mismatch blocked.

## Scope

ML-M3 validates the frozen Q2 hardware-matched model across this reference chain:

```text
PyTorch FP32
-> FP16-weight PyTorch
-> hardware-aware bit model
-> H8 staged RTL
-> H9 interleaved RTL
```

The RTL boundary is exactly one Transformer layer:

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

Software remains responsible for token IDs, BPE tokenizer, token embedding,
learned absolute position embedding, final RMSNorm, tied LM head, logits, and
token selection.

## Frozen Model

```text
model_name=VEDA-HWLM-1L64-Q2
num_layers=1
D_MODEL=64
N_HEAD=8
N_KV_HEAD=8
D_HEAD=8
D_FFN=256
context_length=128
vocab_size=2048
norm=RMSNorm
activation=ReLU
bias=false
position=learned absolute, software-side
weight_tying=true
dropout=0
```

## Frozen Artifacts

```text
checkpoint=D:/IC_Workspace/VEDA_artifacts/ml_q2/benchmark/checkpoints/VEDA-HWLM-1L64-Q2.pt
checkpoint_sha256=68b520f1322c79e568c39115809b8d623e21478af1662658cf997bf174cc9214
tokenizer=D:/IC_Workspace/VEDA_artifacts/ml_m2/formal/tokenizer/tokenizer.json
tokenizer_sha256=72c4100b9c923f8fc89ea563cdf18743742b87ad7cda6732606b61f50f290a1a
export_dir=D:/IC_Workspace/VEDA_artifacts/ml_q2/benchmark/exports
trace_dir=D:/IC_Workspace/VEDA_artifacts/ml_q2/benchmark/traces
ml_m3_artifact_root=D:/IC_Workspace/VEDA_artifacts/ml_m3
```

## Hardware Baseline

Hardware sources are read-only from:

```text
D:/IC_Workspace/VEDA
tag=hw-h9-sfu-pe-interleaving-thesis-accepted
commit=9e0b4c9ba42356ee68e489e99cc5cf64e94f607e
branch=hw/h9-sfu-pe-interleaving
```

ML-M3 must not modify hardware files, hardware branches, RTL, PDK collateral, or
hardware reports.

## Mandatory Gate

One-token H9 and H8 real RTL must pass bit-exact comparison against the
hardware-aware bit model before multi-token length 2/8/16 runs. The current
M3 run is blocked at this gate:

```text
CHECK_FAIL layer token=0 dim=1 got=3d4a2576 expected=3d4a2572
```
