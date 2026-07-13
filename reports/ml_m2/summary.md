# ML-M2 Summary

## Status

MODEL STAGE M2 PASS.

FORMAL TRAINING COMPLETE.

## Formal Model

```text
architecture=decoder-only causal LM
num_layers=1
d_model=64
n_head=8
n_kv_head=8
d_head=8
d_ffn=256
norm=RMSNorm
activation=ReLU
bias=false
position=learned absolute, software-side
context_length=128
tokenizer=deterministic BPE
vocab_size=2048
weight_tying=true
dropout=0
```

## Formal Data

```text
dataset=TinyStories
source=https://huggingface.co/datasets/roneneldan/TinyStories
source_commit=f54c09fd23315a6f9c86f9dc80f725de7d8f9c64
license=cdla-sharing-1.0
train_stories=100000
validation_stories=10000
train_packed_sequences=303287
validation_packed_sequences=28370
tokenizer_training_docs=10000 train-split prefix
training_tokens=116462208 over 3 epochs
```

## Formal Results

```text
gpu=NVIDIA GeForce RTX 5080
driver=576.88
torch=2.10.0.dev20251204+cu128
cuda_runtime=12.8
bf16=true
batch_size=1024
effective_batch_tokens=131072
epochs=3
steps=891
elapsed_seconds=121.449
tokens_per_second=958937.665
initial_train_loss=7.68487024307251
best_validation_loss=3.300639416490282
perplexity=27.129980732808296
no_nan_inf=true
checkpoint_sha256=cfaae278aa7fccd903b3b65041bce1b4dd91410ce3cdeacfb50e5b2b6ca933c8
```

The one-layer model is useful for RTL deployment and numerical validation. Its
generation is functional but repetitive, as expected for a very small one-layer
model.

## Export And Trace

```text
exported_tensor_count=12
export_layout=weight[output_index][input_index]
trace_prompt_lengths=1,2,8,16
trace_node_count_each=40
hardware_aware_top1_agreement=1.0
hardware_aware_top5_overlap=1.0
max_pytorch_vs_hardware_aware_logit_error=0.0021828413009643555
```

## Next Step

Model Stage M3 may start after user approval. ML-M2 did not run real RTL
co-simulation and did not modify Hardware Stage H8.
