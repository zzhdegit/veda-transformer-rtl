# ML-M2 Summary

## Status

MODEL STAGE M2 PIPELINE PASS.

FORMAL TRAINING PENDING.

## Model

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
context_length=128 formal, 64 smoke
tokenizer=BPE
vocab_size=2048 formal, 256 smoke
weight_tying=true
```

## Dataset

Formal dataset is TinyStories, license `cdla-sharing-1.0`, recorded from the
Hugging Face dataset card on 2026-07-13. Full data is not committed.

Smoke dataset is the built-in fixture under `ml/data/fixtures.py`.

## Results

```text
smoke_initial_loss=44.47267532348633
smoke_final_loss=15.681257247924805
smoke_validation_loss=14.336987495422363
smoke_perplexity=1684513.7347844446
formal_training=PENDING
exported_tensors=12
trace_nodes=35
hardware_aware_max_abs_error=0.0028839111328125
```

## Next Step

Do not start Model Stage M3 automatically. With user approval, Model Stage M3
can consume ML-M2 pipeline outputs for PyTorch / bit model / RTL co-simulation.

