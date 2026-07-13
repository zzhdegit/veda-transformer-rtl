# ML-Q1 EOS Audit

```text
train_eos_targets=300000
train_total_targets=120840653
train_eos_ratio=0.0024826081935316324
validation_eos_targets=10000
validation_eos_ratio=0.0027537967544049025
pad_eos_confusion=False
eos_is_ignore_index=False
eos_examples=1000
eos_top1_accuracy=0.082
eos_top5_accuracy=0.994
eos_average_probability=0.07064334366744152
eos_average_rank=3.028
non_terminal_eos_false_positive_rate=0.0
```

Generation EOS rates:

```json
{
  "128": {
    "average_generated_length": 118.3,
    "effective_context_capped": true,
    "eos_rate": 0.0,
    "requested_max_new_tokens": 128
  },
  "256": {
    "average_generated_length": 118.3,
    "effective_context_capped": true,
    "eos_rate": 0.0,
    "requested_max_new_tokens": 256
  },
  "48": {
    "average_generated_length": 48.0,
    "effective_context_capped": true,
    "eos_rate": 0.0,
    "requested_max_new_tokens": 48
  }
}
```
