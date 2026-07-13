# Stage 8 Numeric Comparison

## Result

PAPER_ARRAY bit model and RTL are bit-exact for the covered Stage 8 cases.
PAPER_ARRAY and legacy bit models are also bit-exact for current D8/D16
Attention mappings.

## Model Comparison

Command:

```text
python model/attention/paper_attention_reference.py
```

Result:

```text
paper_attention_output_bit_exact=True
max_abs_error=0.0
mae=0.0
rmse=0.0
relative_l2=0.0
cosine_similarity=1.0
max_ulp=0
argmax_match=True
```

## RTL Comparison

The Stage 8D RTL simulations use legacy-generated golden vectors. Since the
current paper-array adapter preserves the legacy add order for covered D8/D16
tiles, these tests are a bit-exact PAPER_ARRAY RTL versus PAPER_ARRAY model
check and a bit-exact PAPER_ARRAY versus legacy check for the same cases.

Covered comparison points:

| Point | Status |
|---|---|
| QK raw score | bit-exact |
| scaled score | bit-exact |
| Softmax probability | bit-exact |
| sV output | bit-exact |
| final head output | bit-exact |
| final MHA output | bit-exact |
| full transformer output | bit-exact for covered vectors |
| valid sequence length | bit-exact |
| status/invalid | bit-exact |
| metadata | bit-exact |

## Add-Order Note

The current D8/D16 adapter uses the same effective FP32 reduction order as the
legacy PE tile path. If a future stage uses more of the 128-cell array in one
transaction and changes add order, bit-exact legacy comparison is no longer
assumed. Future reports must include error statistics and Attention ranking
checks.
