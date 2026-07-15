# ML-M3 Reference Chain

Reference 0 is PyTorch FP32, Reference 1 is FP16-weight PyTorch, and Reference 2 is the existing hardware-aware bit model.

| Case | FP32 vs FP16 max abs | FP32 vs bit max abs | FP16 vs bit max abs | Bit top-1 agreement |
|---|---:|---:|---:|---:|
| len_1 | 0.0102935 | 0.0158553 | 0.0116825 | 1.000 |
| len_2 | 0.0102961 | 0.015852 | 0.0116782 | 1.000 |
| len_8 | 0.029751 | 0.0346479 | 0.0468359 | 1.000 |
| len_16 | 0.0507717 | 0.0366874 | 0.0555696 | 1.000 |
| len_32 | 0.0507717 | 0.0395622 | 0.0555696 | 1.000 |
