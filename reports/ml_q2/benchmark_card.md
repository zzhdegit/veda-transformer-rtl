# VEDA-HWLM-1L64-Q2 Benchmark Card

VEDA-HWLM-1L64-Q2 is an internal hardware benchmark model, not a general chat
model. It preserves the ML-M2/Q1 one-layer hardware-matched architecture:
decoder-only, RMSNorm, standard MHA, ReLU FFN, no bias, learned absolute
software-side position embedding, context 128, BPE-2048, tied embeddings.

```text
final_candidate=q2_from_scratch
checkpoint=D:\IC_Workspace\VEDA_artifacts\ml_q2\benchmark\checkpoints\VEDA-HWLM-1L64-Q2.pt
sha256=68b520f1322c79e568c39115809b8d623e21478af1662658cf997bf174cc9214
train_stories=2109489
train_tokens=842438406
validation_loss=1.9365209649992428
holdout_loss=1.8095625025135131
export_dir=D:\IC_Workspace\VEDA_artifacts\ml_q2\benchmark\exports
trace_dir=D:\IC_Workspace\VEDA_artifacts\ml_q2\benchmark\traces
```

Known limits: one transformer layer, `d_model=64`, context 128, modest generation
quality, and EOS generation is not guaranteed under greedy decoding. Intended
use is Model Stage M3 PyTorch / bit model / real RTL co-simulation.
