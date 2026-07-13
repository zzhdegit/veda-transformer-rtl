# ML-M2 Quantization and Hardware-Aware Analysis

## Formal Checkpoint

```text
checkpoint=D:/IC_Workspace/VEDA_artifacts/ml_m2/formal/checkpoints/ml_m2_formal_best.pt
checkpoint_sha256=cfaae278aa7fccd903b3b65041bce1b4dd91410ce3cdeacfb50e5b2b6ca933c8
export=D:/IC_Workspace/VEDA_artifacts/ml_m2/formal/exports/fp16_best
export_tensor_count=12
```

The FP16 export uses NumPy/PyTorch FP32-to-FP16 conversion for artifact files.
The hardware-aware path uses the accepted repository Stage 7 bit model and its
FP16/FP32 conversion helpers.

## PyTorch vs FP16-Weight PyTorch

| Prompt Len | Max Abs Error | RMSE | Cosine | Top-1 | Top-5 | First Diff |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0009844303 | 0.0003954091 | 0.99999994 | 1.0 | 1.0 | -1 |
| 2 | 0.0010310411 | 0.0003320868 | 1.00000000 | 1.0 | 1.0 | -1 |
| 8 | 0.0012482405 | 0.0003038008 | 0.99999976 | 1.0 | 1.0 | -1 |
| 16 | 0.0012480021 | 0.0002847809 | 0.99999976 | 1.0 | 1.0 | -1 |

## PyTorch vs Hardware-Aware Stage 7 Path

| Prompt Len | Max Abs Error | RMSE | Cosine | Top-1 | Top-5 | First Diff |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0021826029 | 0.0008255960 | 0.99999994 | 1.0 | 1.0 | -1 |
| 2 | 0.0021828413 | 0.0005916748 | 1.00000000 | 1.0 | 1.0 | -1 |
| 8 | 0.0021828413 | 0.0003653573 | 1.00000000 | 1.0 | 1.0 | -1 |
| 16 | 0.0021828413 | 0.0003469018 | 0.99999976 | 1.0 | 1.0 | -1 |

## Layer And KV Cache Error

Worst observed values across prompt lengths:

```text
layer_output_max_abs_error=0.0002460405230522156
k_cache_max_abs_error=0.0009813308715820312
v_cache_max_abs_error=0.0004578828811645508
```

Trace manifests include per-token KV cache history, valid sequence length,
head index order, and dimension index order. This is sufficient for Model Stage
M3 to build RTL co-simulation vectors without changing the ML-M2 model.
