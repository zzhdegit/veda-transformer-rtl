# ML-M2 Acceptance Audit

## Result

MODEL STAGE M2 PASS.

FORMAL TRAINING COMPLETE.

## Git Isolation

```text
worktree=D:/IC_Workspace/VEDA_ml_m2
branch=ml/m2-hardware-matched-model
remote=origin https://github.com/zzhdegit/veda-transformer-rtl.git
hardware_worktree=D:/IC_Workspace/VEDA
hardware_branch=stage8-paper-pe-array
```

ML-M2 changed only model-line code, scripts, reports, `Makefile` ML targets,
and model-line state/handoff files. It did not modify RTL, accepted bit-model
directories, or Hardware Stage H8 documents.

## Environment

```text
gpu=NVIDIA GeForce RTX 5080
driver=576.88
vram=16303 MiB
python=3.10.19 in deepsc_new for GPU run
torch=2.10.0.dev20251204+cu128
torch_cuda_runtime=12.8
compute_capability=12.0
bf16_supported=true
cuda_smoke=pass
```

The base Anaconda environment remains CPU-only for pytest. GPU training used
the existing CUDA 12.8 PyTorch environment.

## Numeric Audit

The old smoke loss anomaly was caused by unstable default initialization for
the tied embedding/LM head and by PAD labels participating in the loss. Fixes:

- ML-M2 initializer now uses normal `std=0.02` and zeros the PAD embedding row.
- RMSNorm gamma is initialized to 1.
- LM labels replace PAD with `-100`.
- Regression tests cover label shift, PAD ignore, vocab range, logits layout,
  tied LM head, initial loss scale, and single-batch overfit.

```text
untrained_loss_vocab_256=5.566272735595703
log_vocab_256=5.545177444479562
untrained_loss_vocab_2048=7.611394882202148
log_vocab_2048=7.6246189861593985
single_batch_initial_loss=4.178725242614746
single_batch_final_loss=0.41808220744132996
single_batch_top1=0.921875
reload_max_abs_diff=0.0
```

## Formal Data And Training

```text
dataset=TinyStories
source_commit=f54c09fd23315a6f9c86f9dc80f725de7d8f9c64
license=cdla-sharing-1.0
train_stories=100000
validation_stories=10000
tokenizer=BPE-2048, special IDs PAD=0 BOS=1 EOS=2 UNK=3
tokenizer_training_docs=10000 train-split prefix
train_packed_sequences=303287
validation_packed_sequences=28370
batch_size=1024
effective_batch_tokens=131072
epochs=3
steps=891
elapsed_seconds=121.4491955000558
initial_train_loss=7.68487024307251
best_validation_loss=3.300639416490282
perplexity=27.129980732808296
no_nan_inf=true
```

## Export And Trace

```text
best_checkpoint_sha256=cfaae278aa7fccd903b3b65041bce1b4dd91410ce3cdeacfb50e5b2b6ca933c8
last_checkpoint_sha256=968ee2d583493a816b860b05568f91c6a4ff948e0ffce82a597c151eb927fb2a
exported_tensor_count=12
export_manifest_sha256=5cdff9e7332daa2aea0d452478286ad9a43cd1e0bed33493fa2e8770885708c4
trace_prompt_lengths=1,2,8,16
trace_node_count_each=40
max_pytorch_vs_hardware_aware_logit_error=0.0021828413009643555
top1_agreement=1.0
top5_overlap=1.0
```

## Test Results

`make`, `mingw32-make`, and `nmake` are not installed on this Windows host.
The Makefile targets call the Python runners below, and these were run directly:

```text
python scripts/ml/run_ml_m2_numeric_audit.py: PASS
python scripts/ml/run_ml_m2_all_tests.py: PASS
data/tokenizer tests: 8 passed
architecture tests: 11 passed
smoke training tests: 1 passed
export/trace tests: 4 passed
training numerics tests: 7 passed
gpu check: PASS in deepsc_new CUDA environment
formal data: PASS
formal train: PASS
formal eval: PASS
formal export: PASS
```

## Exit Conditions

All ML-M2 Formal exit conditions are met. Model Stage M3 is not started by this
audit.
