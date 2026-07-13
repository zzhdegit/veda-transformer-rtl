# ML-M2E Report: Formal One-Layer Training

## Result

Formal TinyStories training is PENDING in this environment because no CUDA GPU
is available.

This is not a full Model Stage M2 acceptance pass. The valid status after ML-M2
pipeline closure is:

```text
MODEL STAGE M2 PIPELINE PASS
FORMAL TRAINING PENDING
```

## Environment Check

```text
torch.cuda.is_available=false
torch.cuda.device_count=0
cuda_device_name=no cuda
```

## Formal Target

```text
dataset=TinyStories
subset=100000 examples
num_layers=1
d_model=64
n_head=8
d_head=8
d_ffn=256
context=128
vocab=2048
activation=ReLU
norm=RMSNorm
bias=none
position=learned absolute, software-side
seed=20260713
```

## Reproduction Command

After placing TinyStories files under `VEDA_ML_DATA_ROOT` and setting
`VEDA_ML_ARTIFACT_ROOT`:

```bash
python -m ml.training.train --mode formal --config ml/configs/ml_m2_formal.json --output-dir %VEDA_ML_ARTIFACT_ROOT%/formal
```

The current `train.py` blocks formal mode until a GPU-backed run is explicitly
started in the intended artifact environment.

