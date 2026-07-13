# ML-M2 Reproducibility

## Seeds

Default seed:

```text
20260713
```

The ML-M2 path sets Python, NumPy, and Torch seeds. Torch deterministic
algorithms are requested with `warn_only=True`.

## Artifact Roots

Formal run environment:

```text
VEDA_ML_DATA_ROOT=D:/IC_Workspace/VEDA_artifacts/datasets
VEDA_ML_ARTIFACT_ROOT=D:/IC_Workspace/VEDA_artifacts/ml_m2
VEDA_HF_CACHE=D:/IC_Workspace/VEDA_artifacts/hf_cache
```

These paths are local artifact storage and are not Git worktrees.

## Dataset Source

```text
dataset=TinyStories
source=https://huggingface.co/datasets/roneneldan/TinyStories
train_url=https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt
validation_url=https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt
source_commit=f54c09fd23315a6f9c86f9dc80f725de7d8f9c64
license=cdla-sharing-1.0
access_date=2026-07-13
```

Formal subsets are deterministic file prefixes:

```text
train_stories=100000
validation_stories=10000
```

## Reproduction Commands

GPU check:

```powershell
$env:VEDA_ML_DATA_ROOT='D:/IC_Workspace/VEDA_artifacts/datasets'
$env:VEDA_ML_ARTIFACT_ROOT='D:/IC_Workspace/VEDA_artifacts/ml_m2'
$env:VEDA_HF_CACHE='D:/IC_Workspace/VEDA_artifacts/hf_cache'
& 'C:\Users\zzh\anaconda3\envs\deepsc_new\python.exe' scripts\ml\run_ml_m2_gpu_check.py
```

Formal data:

```powershell
$env:VEDA_ML_M2_TRAIN_STORIES='100000'
$env:VEDA_ML_M2_VALIDATION_STORIES='10000'
$env:VEDA_ML_M2_TOKENIZER_DOCS='10000'
python scripts\ml\run_ml_m2_formal_data.py
```

Formal train:

```powershell
& 'C:\Users\zzh\anaconda3\envs\deepsc_new\python.exe' -m ml.training.formal_train --root D:/IC_Workspace/VEDA_artifacts/ml_m2/formal --batch-size 1024 --epochs 3 --validation-interval 100
```

Formal eval/export:

```powershell
& 'C:\Users\zzh\anaconda3\envs\deepsc_new\python.exe' scripts\ml\run_ml_m2_formal_eval.py
& 'C:\Users\zzh\anaconda3\envs\deepsc_new\python.exe' scripts\ml\run_ml_m2_formal_export.py
```

Regression:

```powershell
python scripts\ml\run_ml_m2_numeric_audit.py
python scripts\ml\run_ml_m2_all_tests.py
```

## Host Make Availability

`make`, `mingw32-make`, and `nmake` are not installed on this Windows host. The
Makefile targets are present and call the Python runner commands above.
