# Hardware Stage H9 Numerical Results

Status: PASS for host H9/H8 bit-model comparison and implemented RTL
bit-exact matrices.

Host/model command:

```text
python3 scripts/sim/run_hw_h9_tests.py
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

RTL numerical evidence:

- H9 single-head D_HEAD 8, 16, and 64: PASS, output lanes matched expected FP32
  values.
- H9 matched A/B D_HEAD 8, 16, and 64 at seq 1, 2, 8, 16, 32, and 64: PASS.
- H9 multi-head interleaved H1/D8, H2/D8, H4/D8, H2/D16, and H1/D64: PASS
  against the Stage 5 reference vectors.
- H9 full-layer interleaved H1/D8, H2/D8, H2/D8 two-token, H4/D8, and H2/D16:
  PASS against the Stage 7D reference vectors.
- H9 direct datapath random backpressure: 20 fixed seeds PASS with bit-exact
  FP32 output checks after randomized source/output/done stalls.
- H9 independent multi-head random backpressure: 24 fixed-seed runs PASS with
  bit-exact output checks against the accepted Stage 5 reference vectors.

Limit: full-layer internal multi-endpoint random-backpressure numerical
coverage remains a deferred strict IP-grade verification enhancement. It is not
claimed as closed by the undergraduate thesis acceptance baseline.
