# ML-Q1 Generation Comparison

Full prompt-suite JSON artifacts:

```text
baseline=D:/IC_Workspace/VEDA_artifacts/ml_q1/baseline_eval/prompt_suite_m2_baseline.json
candidate=D:/IC_Workspace/VEDA_artifacts/ml_q1/candidate/eval/prompt_suite_candidate.json
```

Greedy:

```text
baseline_loops=833
candidate_loops=753
baseline_collapse_count=10
candidate_collapse_count=4
```

Sampling temperature 1.0:

```text
baseline_distinct1=0.475
candidate_distinct1=0.5208333333333334
baseline_distinct2=0.851063829787234
candidate_distinct2=0.902127659574468
```

Temperature can improve diversity at decode time, but ML-Q1 acceptance is based
on trained checkpoint metrics, not decoding tricks alone.
