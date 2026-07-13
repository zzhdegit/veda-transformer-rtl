# ML-Q1 Baseline vs Candidate

```text
baseline_checkpoint=D:\IC_Workspace\VEDA_artifacts\ml_m2\formal\checkpoints\ml_m2_formal_best.pt
candidate_checkpoint=D:\IC_Workspace\VEDA_artifacts\ml_q1\candidate\checkpoints\ml_q1_candidate_best.pt
candidate_sha256=fbb8b1815d03a0c9fb3cb1559c3cb6942038e7174d7d434ef57f79cb492994da
baseline_validation_loss=3.3014952083935727
candidate_validation_loss=3.1589355026367603
baseline_holdout_loss=3.3407311116167833
candidate_holdout_loss=3.116297947013361
baseline_validation_ppl=27.153208288215172
candidate_validation_ppl=23.5455184416793
baseline_holdout_ppl=28.239765580499753
candidate_holdout_ppl=22.56269654099442
quality_improvement_pass=True
```

Greedy loop comparison:

```text
baseline_ngram_loops=833
candidate_ngram_loops=753
baseline_distinct1=0.06041666666666666
candidate_distinct1=0.09166666666666665
baseline_distinct2=0.08297872340425533
candidate_distinct2=0.14255319148936169
```

EOS comparison:

```text
baseline_eos_avg_rank=3.028
candidate_eos_avg_rank=3.042
baseline_eos_avg_prob=0.07064334366744152
candidate_eos_avg_prob=0.14027221381517302
```

Hardware-aware comparison:

```json
{
  "prompt": "Once upon a time",
  "pytorch_vs_fp16_weight": {
    "cosine_similarity": 1.0,
    "first_differing_token": -1,
    "max_abs_error": 0.0019259452819824219,
    "mean_abs_error": 0.00025563209783285856,
    "rmse": 0.00033731196890585124,
    "top1_agreement": 1.0,
    "top5_overlap": 1.0
  },
  "pytorch_vs_hardware_aware": {
    "cosine_similarity": 0.9999998807907104,
    "first_differing_token": -1,
    "max_abs_error": 0.0030088424682617188,
    "mean_abs_error": 0.00043762981658801436,
    "rmse": 0.0006483537144958973,
    "top1_agreement": 1.0,
    "top5_overlap": 1.0
  }
}
```
