# ML-M2D Report: Smoke Training

## Result

ML-M2D adds a CPU smoke training flow. The smoke flow trains the one-layer
hardware-matched model on the built-in fixture, saves an artifact checkpoint,
reloads it, runs greedy generation, and verifies incremental KV decode against
full-sequence forward.

## Scope

The smoke checkpoint is written under the artifact root and is not committed to
Git. The repository commits only code, tests, and this report.

## Smoke Configuration

```text
num_layers=1
d_model=64
n_head=8
d_head=8
d_ffn=256
context=64 by default; tests may shrink to 32
vocab=256
dataset=builtin fixture
device=CPU
dtype=FP32
```

## Acceptance Checks

- initial and final loss are recorded;
- final loss decreases in the deterministic smoke test;
- no NaN/Inf;
- checkpoint save/reload;
- reload logits match original logits;
- greedy generation runs;
- full-sequence and incremental KV outputs match.

## Recorded CPU Smoke Run

Command:

```bash
python -m ml.training.train --config ml/configs/ml_m2_smoke.json --output-dir build/ml_m2_artifacts/smoke --mode smoke
```

Result:

```text
device=cpu
steps=12
epochs=19.2
train_examples=5
validation_examples=1
elapsed_seconds=0.043
initial_train_loss=44.47267532348633
final_train_loss=15.681257247924805
validation_loss=14.336987495422363
perplexity=1684513.7347844446
no_nan_inf=true
checkpoint=build/ml_m2_artifacts/smoke/checkpoints/ml_m2_smoke_last.pt
checkpoint_sha256=f8aad2cd9cb4cc68f48fb532ea0689677ede660fbb62c8b2d3d69fa1717c561f
```

The high validation perplexity is expected for a tiny fixture run; smoke
training only proves the pipeline, loss descent, reload, generation, and KV
decode mechanics.
