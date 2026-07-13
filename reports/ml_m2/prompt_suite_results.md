# ML-M2 Prompt Suite Results

## Prompt Suite

The fixed prompt suite is tracked in:

```text
ml/evaluation/ml_m2_prompt_suite.json
```

Prompts:

1. Once upon a time
2. There was a little girl named
3. The dog went to the
4. One day, Tom found a
5. Lily was afraid because
6. The red ball was
7. In the forest, the rabbit
8. The boy said to his mother
9. A small bird could not
10. After the rain stopped

Full JSON output:

```text
D:/IC_Workspace/VEDA_artifacts/ml_m2/formal/interactive_traces/prompt_suite_results.json
```

## Decode Variants

Each prompt was evaluated with:

- greedy;
- temperature 0.7;
- temperature 0.8;
- temperature 1.0;
- top-k 20;
- top-k 40;
- top-p 0.9;
- default interactive sampling with repetition penalty 1.1.

## Aggregate Metrics

| Variant | Repeat | Distinct-1 | Distinct-2 | EOS Rate | Special Ratio | Entropy | N-gram Loops | Collapse Count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 0.0043 | 0.0604 | 0.0830 | 0.0000 | 0.0000 | 3.4207 | 833 | 10 |
| temperature_0_7 | 0.0043 | 0.3500 | 0.6532 | 0.0000 | 0.0000 | 2.3804 | 253 | 0 |
| temperature_0_8 | 0.0000 | 0.3896 | 0.7234 | 0.0000 | 0.0000 | 2.6432 | 196 | 0 |
| temperature_1_0 | 0.0000 | 0.4750 | 0.8511 | 0.0000 | 0.0000 | 3.3635 | 105 | 0 |
| top_k_20 | 0.0000 | 0.2333 | 0.4340 | 0.0000 | 0.0000 | 1.3845 | 422 | 0 |
| top_k_40 | 0.0000 | 0.2729 | 0.5149 | 0.0000 | 0.0000 | 1.6415 | 355 | 0 |
| top_p_0_9 | 0.0000 | 0.3875 | 0.7191 | 0.0000 | 0.0000 | 2.3354 | 199 | 0 |
| sample_default_penalty | 0.0000 | 0.2896 | 0.5489 | 0.0000 | 0.0000 | 1.5773 | 323 | 0 |

## Repetition Findings

- Greedy decoding collapses on every prompt. The dominant failure is a repeated
  space-plus-common-word pattern, not special-token leakage.
- Temperature sampling improves diversity substantially. Temperature 1.0 gives
  the best distinct metrics and lowest n-gram loop count, but output grammar is
  still weak.
- Top-k 20 and top-k 40 are safer than greedy but still repetitive because the
  model distribution is highly peaked.
- Top-p 0.9 is close to temperature 0.8 in diversity and avoids single-token
  collapse in this suite.
- Default repetition penalty removes the strict greedy collapse, but does not
  solve local phrase loops.
- No variant generated PAD/BOS/UNK leakage. EOS was not produced within 48 new
  tokens in any variant.

## Example Outputs

Greedy:

```text
Once upon a time   a a a a a a a a a a a a a a a a a a a a a a a
```

Temperature 0.8:

```text
Once upon a time was a his day, there was a was to who ap Timmy her to a to to she alstp. was Tom looking get
```

Top-p 0.9:

```text
Once upon a time was a his day, there was a was to who a Timmy her to a to to she alstp. was Tom looking get
```

## Incremental KV Check

Full-sequence and incremental KV decoding were compared for all 10 prompts:

```text
allclose=true
max_abs_error=3.337860107421875e-06
```

This supports using the checkpoint for Model Stage M3 software/RTL preparation,
but this task did not run real RTL.
