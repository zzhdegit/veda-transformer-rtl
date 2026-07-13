# Hardware Stage H9 Numeric Comparison

H9 changes paper Attention mapping and schedule. It preserves FP16/FP32 arithmetic boundaries and Stage 3 softmax arithmetic.

Model command:

```text
python -m pytest tb/model/test_hw_h9_interleaved_attention.py
```

Host result:

```text
6 passed
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

RTL smoke result:

```text
tb_h9_single_head D_HEAD=8: PASS
tb_h9_single_head D_HEAD=16: PASS
tb_h9_single_head D_HEAD=64: PASS
```

Full H9 RTL-vs-H9-bit-model acceptance remains open for multi-head, full-layer,
long sequence, reset, random backpressure, and cache-full configurations.
