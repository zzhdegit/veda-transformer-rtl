# Hardware Stage H9 Cache-full Results

Status: coverage entry implemented; not executed in the current environment.

The Stage 5 generation vector stream always emits `MAX_SEQ_LEN + 1` tokens.
The final token is expected to report the existing cache-full behavior from
the Stage 5 reference:

- no new K/V commit;
- `valid_seq_len` does not increase;
- done/status follows the existing cache-full semantics;
- no output tile is expected for the failed extra token;
- reset after the sequence returns the generator to a clean state.

H9-specific configured cache-full runs:

| Run | N_HEAD | D_HEAD | MAX_SEQ_LEN |
|---|---:|---:|---:|
| h9_multi_head_h1_d8_seq8 | 1 | 8 | 8 |
| h9_multi_head_h2_d8_seq8 | 2 | 8 | 8 |
| h9_multi_head_h4_d8_seq8 | 4 | 8 | 8 |
| h9_multi_head_h2_d16_seq8 | 2 | 16 | 8 |
| h9_multi_head_h1_d64_seq8 | 1 | 64 | 8 |
| h9_sequence_cache_full_h1_d8_seq32 | 1 | 8 | 32 |

Current execution result:

```text
vcs: NOT FOUND
result=FAIL
```

Because VCS is unavailable in this environment, cache-full H9 closure remains
open.
