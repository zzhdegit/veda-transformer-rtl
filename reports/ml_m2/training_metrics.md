# ML-M2 Training Metrics

## Numeric Audit

```text
untrained_loss_vocab_256=5.566272735595703
log_vocab_256=5.545177444479562
untrained_loss_vocab_2048=7.611394882202148
log_vocab_2048=7.6246189861593985
single_batch_overfit_initial_loss=4.178725242614746
single_batch_overfit_final_loss=0.41808220744132996
single_batch_overfit_top1=0.921875
single_batch_reload_max_abs_diff=0.0
```

## Formal TinyStories Training

```text
device=NVIDIA GeForce RTX 5080
dtype=bf16
train_stories=100000
validation_stories=10000
train_packed_sequences=303287
validation_packed_sequences=28370
batch_size=1024
effective_batch_tokens=131072
epochs=3
steps=891
elapsed_seconds=121.4491955000558
tokens_per_second=958937.6654203238
peak_allocated_vram_bytes=3932451328
peak_reserved_vram_bytes=5083496448
gpu_utilization_percent_at_end=76
untrained_validation_loss=7.684831942830767
initial_train_loss=7.68487024307251
final_train_loss=3.2740631103515625
best_validation_loss=3.300639416490282
final_validation_loss=3.300639416490282
perplexity=27.129980732808296
last_grad_norm=0.8992450833320618
no_nan_inf=true
```

## Validation History

| Step | Epoch | Train Loss | Validation Loss | Perplexity |
|---:|---:|---:|---:|---:|
| 1 | 0.003 | 7.684870 | 7.678069 | 2160.444 |
| 100 | 0.337 | 5.316854 | 5.298386 | 200.014 |
| 200 | 0.673 | 3.988019 | 3.977815 | 53.400 |
| 300 | 1.010 | 3.554584 | 3.539315 | 34.443 |
| 400 | 1.347 | 3.438623 | 3.436622 | 31.082 |
| 500 | 1.684 | 3.375443 | 3.377590 | 29.300 |
| 600 | 2.020 | 3.341717 | 3.342755 | 28.297 |
| 700 | 2.357 | 3.328204 | 3.321539 | 27.703 |
| 800 | 2.694 | 3.315238 | 3.308843 | 27.353 |
| 891 | 3.000 | 3.274063 | 3.300639 | 27.130 |

## Generation

Greedy generation runs from the formal best checkpoint and does not emit only
special tokens. Samples are stored at:

```text
D:/IC_Workspace/VEDA_artifacts/ml_m2/formal/training/generation_samples.json
D:/IC_Workspace/VEDA_artifacts/ml_m2/formal/evaluation/formal_evaluation.json
```

The samples are repetitive, so the model is classified as suitable for RTL
deployment validation and basic generation smoke, not as a quality language
model.
