# Hardware Stage H9 Cache-full Results

Status: PASS for implemented deterministic cache-full RTL coverage.

The Stage 5 generation vector stream emits `MAX_SEQ_LEN + 1` tokens. The final
token is expected to use the existing Stage 5 cache-full behavior:

- no new K/V commit;
- `valid_seq_len` does not increase;
- done/status follows the existing cache-full semantics;
- no output tile is expected for the failed extra token;
- reset before the next clean stream returns the generator to a clean state.

H9 configured cache-full runs:

| Run | N_HEAD | D_HEAD | MAX_SEQ_LEN | Result |
|---|---:|---:|---:|---|
| h9_multi_head_h1_d8_seq8 | 1 | 8 | 8 | PASS |
| h9_multi_head_h2_d8_seq8 | 2 | 8 | 8 | PASS |
| h9_multi_head_h4_d8_seq8 | 4 | 8 | 8 | PASS |
| h9_multi_head_h2_d16_seq8 | 2 | 16 | 8 | PASS |
| h9_multi_head_h1_d64_seq8 | 1 | 64 | 8 | PASS |
| h9_sequence_cache_full_h1_d8_seq32 | 1 | 8 | 32 | PASS |

Long-sequence evidence:

```text
step=step31 seq_before=31 seq_after=32 commit=32 peak_seq=32
step=step32 seq_before=32 seq_after=32 commit=32 peak_seq=32
```

The extra token at full cache did not increase `valid_seq_len` or commit count.
