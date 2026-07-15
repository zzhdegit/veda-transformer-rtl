# Hardware Stage H9 Numeric Comparison

H9 changes paper Attention mapping and schedule. It preserves FP16/FP32 arithmetic boundaries and Stage 3 softmax arithmetic.

Model command:

```text
python -m pytest tb/model/test_hw_h9_interleaved_attention.py
```

Host result:

```text
7 passed
```

The H9 model reports H8 differences with:

- max absolute error;
- MAE;
- RMSE;
- relative L2;
- cosine similarity;
- max ULP;
- argmax/ranking consistency.

Model result:

```text
H9/H8 comparison: bit_exact=True for D_HEAD=8, 16, 64, and 128
max_abs_error=0
mae=0
rmse=0
relative_l2=0
cosine_similarity=1
max_ulp=0
argmax/ranking consistent
```

RTL result:

```text
tb_h9_single_head D_HEAD=8: PASS
tb_h9_single_head D_HEAD=16: PASS
tb_h9_single_head D_HEAD=64: PASS
matched A/B D_HEAD=8/16/64 seq=1/2/8/16/32/64: PASS
multi-head H1/D8, H2/D8, H4/D8, H2/D16, H1/D64: PASS
full-layer H1/D8, H2/D8, H2/D8 two-token, H4/D8, H2/D16: PASS
long sequence/cache-full H1/D8 MAX_SEQ_LEN=32 plus extra token: PASS
```

Full H9 numerical acceptance remains open only for the missing 20-seed random
backpressure and full reset interrupt matrix configurations.
