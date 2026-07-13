# ML-M2 Artifact Manifest

## Smoke Artifacts

Artifacts are local and ignored by Git:

```text
build/ml_m2_artifacts/smoke/checkpoints/ml_m2_smoke_last.pt
sha256=f8aad2cd9cb4cc68f48fb532ea0689677ede660fbb62c8b2d3d69fa1717c561f
build/ml_m2_artifacts/smoke/tokenizer/tokenizer.json
build/ml_m2_artifacts/smoke/generation_samples.json
build/ml_m2_artifacts/smoke/smoke_summary.json
```

## Export And Trace Artifacts

Artifacts are local and ignored by Git:

```text
build/ml_m2_artifacts/export/export_manifest.json
build/ml_m2_artifacts/export/*.hex
build/ml_m2_artifacts/export/*.npy
exported_tensor_count=12
build/ml_m2_artifacts/traces/smoke_trace_manifest.json
trace_node_count=35
```

## Formal Artifacts

```text
status=PENDING
reason=CUDA GPU is not available in this environment.
```
