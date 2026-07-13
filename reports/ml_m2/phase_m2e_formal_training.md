# ML-M2E Report: Formal One-Layer Training

## Result

FORMAL TRAINING COMPLETE.

## Environment

```text
gpu=NVIDIA GeForce RTX 5080
driver=576.88
torch=2.10.0.dev20251204+cu128
cuda_runtime=12.8
bf16_supported=true
cuda_smoke=pass
```

## Data

```text
dataset=TinyStories
source_commit=f54c09fd23315a6f9c86f9dc80f725de7d8f9c64
license=cdla-sharing-1.0
train_stories=100000
validation_stories=10000
tokenizer_training_docs=10000 train-split prefix
train_packed_sequences=303287
validation_packed_sequences=28370
packing_utilization=0.9999974498165104
pad_label_ratio_train=0.0000025759429187477423
unk_ratio=0.0000028335444366871156
```

The target 300000-story expansion remains optional for a later training-quality
run. ML-M2 Formal meets the minimum accepted 100000-story requirement and
produces stable RTL-deployment artifacts.

## Training Configuration

```text
num_layers=1
d_model=64
n_head=8
d_head=8
d_ffn=256
context=128
vocab=2048
activation=ReLU
norm=RMSNorm
bias=none
position=learned absolute, software-side
seed=20260713
dtype=BF16 autocast
optimizer=AdamW
lr=3e-4
minimum_lr=3e-5
betas=0.9,0.95
weight_decay=0.1
grad_clip=1.0
schedule=cosine
warmup=2%
batch_size=1024
epochs=3
```

## Metrics

```text
steps=891
elapsed_seconds=121.4491955000558
tokens_per_second=958937.6654203238
untrained_validation_loss=7.684831942830767
initial_train_loss=7.68487024307251
final_train_loss=3.2740631103515625
best_validation_loss=3.300639416490282
perplexity=27.129980732808296
no_nan_inf=true
best_checkpoint_sha256=cfaae278aa7fccd903b3b65041bce1b4dd91410ce3cdeacfb50e5b2b6ca933c8
```

## Checkpoints

```text
best_checkpoint=D:/IC_Workspace/VEDA_artifacts/ml_m2/formal/checkpoints/ml_m2_formal_best.pt
last_checkpoint=D:/IC_Workspace/VEDA_artifacts/ml_m2/formal/checkpoints/ml_m2_formal_last.pt
metrics=D:/IC_Workspace/VEDA_artifacts/ml_m2/formal/training/formal_training_metrics.json
generation_samples=D:/IC_Workspace/VEDA_artifacts/ml_m2/formal/training/generation_samples.json
```

The checkpoint is suitable for ML-M2 export and Model Stage M3 one-layer
co-simulation preparation. ML-M2 did not start Model Stage M3.
