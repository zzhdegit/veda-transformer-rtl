# ML-Q1 Capacity and Training Audit

```text
train_sample_loss=3.295611025765538
validation_loss=3.3014952083935727
holdout_loss=3.3407311116167833
validation_perplexity=27.153208288215172
holdout_perplexity=28.239765580499753
high_frequency_avg_loss=4.8888117854718445
low_frequency_avg_loss=10.514994423631308
first_16_position_loss=3.274016246199608
last_16_position_loss=3.326100453734398
```

Attribution:

```json
{
  "not_overfitting": true,
  "primary": "undertraining plus fixed one-layer capacity limit",
  "secondary": [
    "decode policy brittleness",
    "EOS imbalance/calibration",
    "simple BPE fragmentation"
  ]
}
```
