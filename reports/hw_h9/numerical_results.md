# Hardware Stage H9 Numerical Results

Status: host bit-model comparison passes; expanded RTL numerical matrix is not
executed in the current environment.

Host/model command:

```text
python scripts/sim/run_hw_h9_tests.py
```

Result:

```text
7 H9 pytest cases passed.
H9 vs H8 compare:
d_head=8   output_bit_exact=True max_abs_error=0.0 max_ulp=0
d_head=16  output_bit_exact=True max_abs_error=0.0 max_ulp=0
d_head=64  output_bit_exact=True max_abs_error=0.0 max_ulp=0
d_head=128 output_bit_exact=True max_abs_error=0.0 max_ulp=0
result=PASS
```

Metrics retained for H9 vs H8 output comparison:

```text
max_abs_error = 0.0
MAE = 0.0
RMSE = 0.0
relative_L2 = 0.0
cosine_similarity = 1.0
max_ULP = 0
attention ranking = match
```

The H9 RTL expanded numerical matrix is wired through the multi-head and
full-layer VCS entries, but current execution is blocked:

```text
vcs: NOT FOUND
result=FAIL
```

Therefore `RTL == H9 bit model` is accepted only for the previously recorded
matched single-head checkpoint, not for the newly added multi-head/full-layer
final-closure matrix.
