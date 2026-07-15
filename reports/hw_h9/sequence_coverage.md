# Hardware Stage H9 Sequence Coverage

Status: PASS for implemented sequence and cache-full RTL coverage; full
randomized stress remains open.

The H9 single-head matched RTL A/B evidence covers:

```text
D_HEAD = 8, 16, 64
seq = 1, 2, 8, 16, 32, 64
```

The H9 multi-head Stage 5 vector generator was run with:

```text
MAX_SEQ_LEN=8  for H1/D8, H2/D8, H4/D8, H2/D16, H1/D64
MAX_SEQ_LEN=32 for H1/D8 long-sequence/cache-full coverage
```

The `MAX_SEQ_LEN=32` Stage 5 stream executes successful sequence lengths 1
through 32 and then one extra cache-full token. This includes the required
irregular points 3, 7, 9, 15, and 31.

Result from `reports/hw_h9/rtl_sim.txt`:

```text
h9_sequence_cache_full_h1_d8_seq32 result=PASS
step=step31 seq_before=31 seq_after=32 commit=32 peak_seq=32
step=step32 seq_before=32 seq_after=32 commit=32 peak_seq=32
```

Limit: sequence coverage is deterministic; broad random backpressure and
20-seed stress are not yet implemented.
