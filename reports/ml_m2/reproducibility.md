# ML-M2 Reproducibility

## Seeds

Default seed:

```text
20260713
```

The training path sets Python, NumPy, and Torch seeds. Torch deterministic
algorithms are requested with `warn_only=True`.

## Artifact Roots

```text
VEDA_ML_DATA_ROOT
VEDA_ML_ARTIFACT_ROOT
VEDA_HF_CACHE
```

When these variables are unset, tests use `build/ml_m2_artifacts/`, which is
ignored by Git.

## Required External Data For Formal Training

```text
TinyStories-train.txt
TinyStories-valid.txt
```

These files must be placed under `VEDA_ML_DATA_ROOT` and must not be committed.

