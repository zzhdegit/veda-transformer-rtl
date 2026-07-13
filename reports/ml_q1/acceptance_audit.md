# ML-Q1 Acceptance Audit

## Result

ML-Q1 QUALITY IMPROVEMENT PASS

```text
baseline_checkpoint_unchanged=True
validation_loss_baseline=3.3014952083935727
validation_loss_candidate=3.1589355026367603
holdout_loss_baseline=3.3407311116167833
holdout_loss_candidate=3.116297947013361
greedy_loops_baseline=833
greedy_loops_candidate=753
candidate_incremental_kv_allclose=True
hardware_aware_top1=1.0
```

ML-Q1 did not run real RTL and did not modify hardware files.

## Tests

```text
python -m py_compile ml/evaluation/q1_quality.py scripts/ml/run_ml_q1_quality.py
PASS

python scripts/ml/run_ml_m2_interactive_tests.py
PASS: 5 pytest cases passed

python -c "<Q1 artifact assertions>"
PASS: baseline SHA, candidate incremental KV, hardware-aware top-1, and quality flag checked

make ml-m2-interactive-test
NOT RUN: make is not installed in this Windows shell; equivalent Python entry passed
```

Forbidden-path check for `rtl/`, legacy bit-model directories, Stage H9/H8
reports/docs, and hardware scripts returned no modified paths.
