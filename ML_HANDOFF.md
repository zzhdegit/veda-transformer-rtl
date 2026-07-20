# Model Handoff

## Current Stage

Model Stage M3: Real-Weight RTL Co-Simulation and Deployment Validation

Short id: ML-M3

## Status

MODEL STAGE M3 PASS.

## Completed In ML-M3

- Confirmed model worktree `D:/IC_Workspace/VEDA_ml_m2` on branch
  `ml/m3-real-rtl-cosim`.
- Confirmed hardware read-only repo `D:/IC_Workspace/VEDA` is clean on
  `hw/h9-real-weight-numeric-repair` at tag
  `hw-h9-real-weight-numeric-repair-accepted`.
- Recomputed Q2 checkpoint SHA256:
  `68b520f1322c79e568c39115809b8d623e21478af1662658cf997bf174cc9214`.
- Recomputed tokenizer SHA256:
  `72c4100b9c923f8fc89ea563cdf18743742b87ad7cda6732606b61f50f290a1a`.
- Audited 12 Q2 export tensors and 8 RTL layer weight mappings.
- Generated real Q2 vectors for lengths 1, 2, 8, 16, and 32 under
  `D:/IC_Workspace/VEDA_artifacts/ml_m3/vectors`.
- Added model-line-only RTL testbench and VCS runner.
- Confirmed VCS/vlogan are available inside Docker container `nailong`.
- Compiled/elaborated the repaired H9 RTL for Q2 parameters:
  `N_HEAD=8`, `D_HEAD=8`, `D_MODEL=64`, `D_FFN=256`, `MAX_SEQ_LEN=128`.
- Ran the core real-weight RTL matrix:
  lengths 1, 2, 8, and 16; H8 staged and H9 interleaved; no-stall and
  deterministic output+done stall modes.
- Confirmed H8 RTL == H9 RTL == hardware-aware bit model for every core output
  bit.
- Ran length32 no-stall as an extended co-simulation case; it passed and does
  not block the core acceptance result.
- Compared 9 real RTL internal node categories per schedule against the
  hardware-aware bit model with zero mismatches.
- Closed software full-vs-incremental reference comparison.
- Closed valid_seq_len, output lane/tile, done, metadata, duplicate output, and
  missing output checks in the model-line testbench.
- Generated H8/H9 cycle tables and Attention/full-layer cycle deltas.
- Ran 3 hybrid next-token cases and one continuous 2-step prediction.
- Ran model-side regression (`ml/tests/test_architecture.py`, 11 tests) and
  M3 Python compile checks.
- Ran forbidden-path and hardware read-only audits.
- Generated final M3 reports under `reports/ml_m3/` and final artifact
  manifests under `D:/IC_Workspace/VEDA_artifacts/ml_m3`.

## ML-M3 Reproduction

```bash
python scripts/ml/run_ml_m3_artifact_audit.py
python scripts/ml/run_ml_m3_vector_generation.py
python scripts/ml/run_ml_m3_vcs.py --length 1 --length 2 --length 8 --length 16 --schedule staged --schedule interleaved --stall-mode none --stall-mode output_done --run-id repair_core_len1_2_8_16
python scripts/ml/run_ml_m3_compare.py
python scripts/ml/run_ml_m3_acceptance.py
```

Extended length32 reproduction:

```bash
python scripts/ml/run_ml_m3_vcs.py --length 32 --schedule staged --schedule interleaved --stall-mode none --run-id repair_len32_no_stall
```

## Not Completed In ML-M3

- No remaining core M3 items.
- Length32 output+done stall is not required by the ML-M3 acceptance standard
  because length32 is an extended case; length32 no-stall passed.

## Dependencies

- Host Python 3.12 environment for model/reference/report scripts.
- Docker Desktop and existing container `nailong` for VCS.
- Synopsys VCS/vlogan and DesignWare simulation libraries inside `nailong`.
- Read-only hardware repository at `D:/IC_Workspace/VEDA` on repair commit
  `a54e608a8dc7e63c7e5dd342f8b893bb1e0b7485`.

## Next-Stage Cautions

- Do not modify the Q2 checkpoint, tokenizer, or exported FP16 weights when
  reproducing M3.
- Do not modify `D:/IC_Workspace/VEDA` from the model branch.
- Do not enter Hardware Stage H10, PDK, STA, P&R, PPA, or physical signoff from
  the ML-M3 flow.
- Preserve DesignWare FP32 add RNE mode `3'b000`; do not restore the old
  `3'b100` behavior.
- If future runs alter vectors, bit widths, ready/valid protocol, K/V layout, or
  model architecture, first update the model stage spec and rerun the full M3
  acceptance flow.

## Previous Stage

Model Stage M2: Hardware-Matched Language Model Training

Short id: ML-M2

## Previous Status

MODEL STAGE M2 PASS.

FORMAL TRAINING COMPLETE.

## Completed

- Model Stage M1 selected the self-trained hardware-matched model as the
  primary deployment path for current RTL.
- ML-M2A froze the training/export contract.
- ML-M2B added dataset and tokenizer pipeline code.
- ML-M2C implemented the hardware-matched PyTorch causal LM.
- ML-M2D closed deterministic CPU smoke training.
- ML-M2 Formal fixed the old high-loss numeric issue before training.
- ML-M2 Formal prepared TinyStories 100000-train / 10000-validation artifacts.
- ML-M2 Formal trained the one-layer model on RTX 5080 with BF16.
- ML-M2F exported formal FP16 weights and hardware-aware traces.
- ML-M2G closed acceptance audit, summary, artifact manifest, project state,
  and handoff.

## Formal Artifacts

```text
artifact_root=D:/IC_Workspace/VEDA_artifacts/ml_m2/formal
best_checkpoint=D:/IC_Workspace/VEDA_artifacts/ml_m2/formal/checkpoints/ml_m2_formal_best.pt
best_checkpoint_sha256=cfaae278aa7fccd903b3b65041bce1b4dd91410ce3cdeacfb50e5b2b6ca933c8
last_checkpoint_sha256=968ee2d583493a816b860b05568f91c6a4ff948e0ffce82a597c151eb927fb2a
export_dir=D:/IC_Workspace/VEDA_artifacts/ml_m2/formal/exports/fp16_best
trace_dir=D:/IC_Workspace/VEDA_artifacts/ml_m2/formal/traces
```

## Key Metrics

```text
gpu=NVIDIA GeForce RTX 5080
torch=2.10.0.dev20251204+cu128
cuda_runtime=12.8
bf16_supported=true
train_stories=100000
validation_stories=10000
batch_size=1024
effective_batch_tokens=131072
epochs=3
steps=891
elapsed_seconds=121.4491955000558
initial_train_loss=7.68487024307251
best_validation_loss=3.300639416490282
perplexity=27.129980732808296
```

## Reproduction Entry Points

Expected Makefile targets exist:

```bash
make ml-m2-unit-test
make ml-m2-data-test
make ml-m2-tokenizer-test
make ml-m2-smoke-test
make ml-m2-export-test
make ml-m2-trace-test
make ml-m2-test
make ml-m2-gpu-check
make ml-m2-numeric-audit
make ml-m2-formal-data
make ml-m2-formal-train
make ml-m2-formal-eval
make ml-m2-formal-export
```

This Windows host does not have `make`, `mingw32-make`, or `nmake` installed,
so the equivalent Python runners were used directly.

## Not Completed

- Real RTL co-simulation.
- Model Stage M3.
- SmolLM2 adaptation.
- Qwen eviction experiments.
- Hardware Stage H8 integration.

## Post-Acceptance ML-Q2 Result

Model Quality Experiment Q2 completed a full-dataset fixed-architecture
benchmark without changing the ML-M2 hardware math contract.

```text
status=ML-Q2 FULL-DATASET BENCHMARK PASS
branch=ml/q2-full-dataset-benchmark
artifact_root=D:/IC_Workspace/VEDA_artifacts/ml_q2
benchmark_checkpoint=D:/IC_Workspace/VEDA_artifacts/ml_q2/benchmark/checkpoints/VEDA-HWLM-1L64-Q2.pt
benchmark_checkpoint_sha256=68b520f1322c79e568c39115809b8d623e21478af1662658cf997bf174cc9214
validation_loss=1.9365209649992428
validation_perplexity=6.934583296916738
holdout_loss=1.8095625025135131
holdout_perplexity=6.10777764135872
export_dir=D:/IC_Workspace/VEDA_artifacts/ml_q2/benchmark/exports
trace_dir=D:/IC_Workspace/VEDA_artifacts/ml_q2/benchmark/traces
```

ML-Q2 did not overwrite the accepted ML-M2 baseline checkpoint or the ML-Q1
candidate checkpoint. It did not run real RTL and did not modify functional RTL
or Hardware Stage H9 files.

## Next-Stage Cautions

- Do not modify RTL or Hardware Stage H8 files from the model branch.
- Do not commit datasets, checkpoints, tokenizer caches, FP16 weight exports,
  large traces, or credentials.
- Model Stage M3 should begin from the formal best checkpoint and compare
  PyTorch, hardware-aware bit model, and real RTL. After ML-Q2, the preferred
  fixed-architecture benchmark input is `VEDA-HWLM-1L64-Q2`.
- Generation quality is intentionally modest; use this model primarily for
  hardware-matched numeric validation and one-layer deployment experiments.
