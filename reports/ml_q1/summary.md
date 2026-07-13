# ML-Q1 Summary

## Result

ML-Q1 QUALITY IMPROVEMENT PASS

ML-Q1 keeps the accepted ML-M2 architecture fixed and evaluates whether
continuing TinyStories training improves quality without changing RTL.

```text
train_stories=300000
training_tokens=241681308
epochs=2
steps=1844
candidate=D:\IC_Workspace\VEDA_artifacts\ml_q1\candidate\checkpoints\ml_q1_candidate_best.pt
candidate_sha256=fbb8b1815d03a0c9fb3cb1559c3cb6942038e7174d7d434ef57f79cb492994da
baseline_validation_loss=3.3014952083935727
candidate_validation_loss=3.1589355026367603
baseline_holdout_loss=3.3407311116167833
candidate_holdout_loss=3.116297947013361
```

Conclusion: fixed one-layer `d_model=64` remains the dominant quality limit.
Further full-dataset training should only be run as ML-Q2 if validation,
holdout, loop, EOS, and hardware-aware metrics justify it.

Tests: Q1 script py_compile passed, ML-M2 interactive Python regression passed
5/5, Q1 artifact assertions passed, and no RTL/Hardware Stage H9 paths were
modified. `make ml-m2-interactive-test` was not runnable because `make` is not
installed in this Windows shell; the equivalent Python command passed.
