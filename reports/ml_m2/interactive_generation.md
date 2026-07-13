# ML-M2 Post-Acceptance Interactive Generation

## Scope

This is a Model Stage M2 post-acceptance evaluation task. It does not start
Model Stage M3, does not run real RTL co-simulation, and does not modify
hardware files.

## Artifact Loading

```text
artifact_root=D:/IC_Workspace/VEDA_artifacts/ml_m2/formal
checkpoint=D:/IC_Workspace/VEDA_artifacts/ml_m2/formal/checkpoints/ml_m2_formal_best.pt
checkpoint_sha256=cfaae278aa7fccd903b3b65041bce1b4dd91410ce3cdeacfb50e5b2b6ca933c8
tokenizer=D:/IC_Workspace/VEDA_artifacts/ml_m2/formal/tokenizer/tokenizer.json
tokenizer_sha256=72c4100b9c923f8fc89ea563cdf18743742b87ad7cda6732606b61f50f290a1a
generation_config=D:/IC_Workspace/VEDA_artifacts/ml_m2/formal/interactive_traces/generation_config.json
```

The loader reads the formal data manifest and formal training metrics to locate
the tokenizer, checkpoint, model config, dataset metadata, and generation
configuration. No tokenizer is retrained and no model is reinitialized.

## Commands

Interactive chat:

```powershell
python scripts/ml/run_ml_m2_chat.py --artifact-root D:/IC_Workspace/VEDA_artifacts/ml_m2/formal
```

Single prompt:

```powershell
python scripts/ml/run_ml_m2_chat.py --artifact-root D:/IC_Workspace/VEDA_artifacts/ml_m2/formal --prompt "Once upon a time, there was a small dog"
```

Next-token:

```powershell
python scripts/ml/run_ml_m2_next_token.py --artifact-root D:/IC_Workspace/VEDA_artifacts/ml_m2/formal --prompt "Once upon a time"
```

Inspect:

```powershell
python scripts/ml/run_ml_m2_inspect.py --artifact-root D:/IC_Workspace/VEDA_artifacts/ml_m2/formal --prompt "Once upon a time"
```

## Greedy Sample

Command:

```powershell
python scripts/ml/run_ml_m2_chat.py --artifact-root D:/IC_Workspace/VEDA_artifacts/ml_m2/formal --prompt "Once upon a time, there was a small dog" --mode greedy --max-tokens 32 --json
```

Result:

```text
prompt_token_ids=[1,230,5,240,5,54,5,263,5,206,5,107,5,54,5,556,5,415]
prompt_tokens=<bos>, Once, " ", upon, " ", a, " ", time,, " ", there, " ", was, " ", a, " ", small, " ", dog
generated_text=Once upon a time, there was a small dog a a a a a a a a a a a a a a a a
generated_token_pattern=[5,54] repeated
kv_cache_length=49
tokens_per_second=2319.109
```

Greedy decoding is deterministic but collapses into a space/`a` loop for this
prompt.

## Sampling Sample

Command:

```powershell
python scripts/ml/run_ml_m2_chat.py --artifact-root D:/IC_Workspace/VEDA_artifacts/ml_m2/formal --prompt "Once upon a time, there was a small dog" --mode sample --temperature 0.8 --top-k 40 --top-p 0.9 --repetition-penalty 1.1 --max-tokens 32 --seed 20260713 --json
```

Result:

```text
generated_text=Once upon a time, there was a small dog was very his day, there was a was to little with to her to a to
kv_cache_length=49
average_entropy=1.4957948327064514
tokens_per_second=1070.220
special_token_leak=false
```

Sampling improves diversity but still produces weak grammar and short-range
loops.

## Next-Token Prediction

Prompt:

```text
Once upon a time
```

Tokenized prompt:

```text
ids=[1,230,5,240,5,54,5,204]
tokens=<bos>, Once, " ", upon, " ", a, " ", time
last_position=7
entropy=0.20416796207427979
special_token_probability=0.0005838120705448091
eos_probability=0.0005838120705448091
```

Top-10:

| Rank | Token ID | Token | Logit | Probability |
|---:|---:|---|---:|---:|
| 1 | 5 | space | 8.456926 | 0.97962111 |
| 2 | 72 | s | 1.872502 | 0.00135357 |
| 3 | 12 | , | 1.817340 | 0.00128092 |
| 4 | 54 | a | 1.643340 | 0.00107635 |
| 5 | 14 | . | 1.624357 | 0.00105611 |
| 6 | 4 | newline | 1.562653 | 0.00099292 |
| 7 | 6 | ! | 1.233900 | 0.00071472 |
| 8 | 2 | `<eos>` | 1.031584 | 0.00058381 |
| 9 | 231 | !" | 0.789208 | 0.00045815 |
| 10 | 189 | 's | 0.531382 | 0.00035403 |

The distribution is very peaked on space, explaining much of the greedy
degeneration.

## Inspect Trace

Inspect output:

```text
D:/IC_Workspace/VEDA_artifacts/ml_m2/formal/interactive_traces/inspect_e286222c229e.json
```

Selected nodes:

```text
layer_input shape=[1,8,64] rms=0.103831 checksum=d5d36415de5ba02d7cab236f03833f2ce93d4c56404eafdc4263c118ac237715
Q shape=[1,8,8,8] rms=0.427710 checksum=c5a286238174b50af4b8f3a8a4aa70c3bab12e0d558e73037242635393df7bb7
K shape=[1,8,8,8] rms=0.868095 checksum=2bfa175e77937558c8e1b4edbee19bde9b6a538402142be0962087514d726d20
V shape=[1,8,8,8] rms=0.351032 checksum=4edb72f4bc5985f87b6a84707bddd1a2d39824fb68b70efba686b0ba5ba55e20
attention_probabilities shape=[1,8,8,8] mean=0.125000 rms=0.220054
layer_output shape=[1,8,64] rms=0.266082 checksum=204c97cad08d660dbf2ef21696abdf4020c79ddc4da29c67fe55be254db8a79a
logits shape=[1,8,2048] min=-7.530244 max=8.456926 rms=4.192602
K_cache shape=[1,8,8,8] checksum=2bfa175e77937558c8e1b4edbee19bde9b6a538402142be0962087514d726d20
V_cache shape=[1,8,8,8] checksum=4edb72f4bc5985f87b6a84707bddd1a2d39824fb68b70efba686b0ba5ba55e20
```

The masked attention score tensor reports `-inf`/`inf` statistics because
future-token positions are intentionally masked.
