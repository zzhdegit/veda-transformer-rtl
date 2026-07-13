# ML-M2 GPU Throughput Benchmark

## Environment

```text
gpu=NVIDIA GeForce RTX 5080
driver=576.88
torch=2.10.0.dev20251204+cu128
cuda_runtime=12.8
dtype=bf16
context_length=128
optimizer=AdamW fused when available
torch_compile=not enabled
```

The eager path was retained because this Windows run did not demonstrate a
separate `torch.compile` improvement with identical backward behavior.

## Batch Sweep

| Batch | Tokens/s | Seq/s | Step Time (s) | Peak Allocated VRAM | GPU Util | Status |
|---:|---:|---:|---:|---:|---:|---|
| 64 | 235874.998 | 1842.773 | 0.034730 | 264237568 | 79% | ok |
| 128 | 593704.070 | 4638.313 | 0.027596 | 508785152 | 83% | ok |
| 256 | 196371.043 | 1534.149 | 0.166868 | 997880320 | 98% | ok |
| 512 | 502425.266 | 3925.197 | 0.130439 | 1976070656 | 98% | ok |
| 1024 | 769143.289 | 6008.932 | 0.170413 | 3932451328 | 98% | ok |

Selected batch:

```text
micro_batch=1024
effective_batch_tokens=131072
selection_reason=highest stable measured tokens/s within the requested 32768-131072 effective-token range
```

Formal training peak reserved VRAM was `5083496448` bytes and peak allocated
VRAM was `3932451328` bytes.
