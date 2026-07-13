# ML-M2 Acceptance Audit

## Result

MODEL STAGE M2 PIPELINE PASS.

FORMAL TRAINING PENDING.

Full `MODEL STAGE M2 PASS` is not claimed because this environment has no CUDA
GPU and formal TinyStories training was not run.

## Git Isolation

```text
worktree=D:/IC_Workspace/VEDA_ml_m2
branch=ml/m2-hardware-matched-model
base=e3b2c14a6af10cccc95f47dfadaeec2d0fc923ad
remote=origin https://github.com/zzhdegit/veda-transformer-rtl.git
```

Hardware Stage H8 worktree remains separate and is not modified by ML-M2.

## Contract Audit

The implemented model matches the accepted Stage 7 contract:

- one layer;
- decoder-only causal LM;
- Pre-Norm;
- RMSNorm;
- standard MHA;
- `N_Q_HEAD = N_KV_HEAD = 8`;
- `D_MODEL = 64`;
- `D_HEAD = 8`;
- `D_FFN = 256`;
- WQ/WK/WV/WO no bias;
- W1/W2 no bias;
- ReLU;
- learned absolute position embedding outside RTL;
- embedding/final norm/LM head outside RTL;
- append-only KV cache;
- no RoPE, no GQA, no SwiGLU.

## Test Results

```text
python scripts/ml/run_ml_m2_all_tests.py: PASS
data/tokenizer tests: 6 passed
architecture tests: 11 passed
smoke training tests: 1 passed
export/trace tests: 4 passed
host make ml-m2-test: unavailable, host make command not installed
```

## Smoke Training

```text
device=cpu
steps=12
initial_train_loss=44.47267532348633
final_train_loss=15.681257247924805
validation_loss=14.336987495422363
perplexity=1684513.7347844446
checkpoint_sha256=f8aad2cd9cb4cc68f48fb532ea0689677ede660fbb62c8b2d3d69fa1717c561f
```

## Export And Trace

```text
exported_tensor_count=12
rtl_layout=weight[output_index][input_index]
trace_node_count=35
hardware_aware_max_abs_error=0.0028839111328125
hardware_aware_top1_agreement=1.0
hardware_aware_top5_overlap=1.0
```

## Exit Conditions

All pipeline exit conditions are met except formal TinyStories training. The
formal status is recorded as PENDING in
`reports/ml_m2/formal_training_status.json`.
