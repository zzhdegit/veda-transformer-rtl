# Hardware Stage H9 Multi-head Results

Status: PASS for the implemented interleaved multi-head RTL matrix.

Command:

```text
make hw-h9-rtl-sim
```

The H9 RTL script compiles the Stage 5 `multi_head_generation_engine` testbench
with:

```text
ATTENTION_PE_ARCH=PAPER_ARRAY
ATTENTION_SCHEDULE=INTERLEAVED
```

Configured runs:

| Run | N_HEAD | D_HEAD | MAX_SEQ_LEN | Result |
|---|---:|---:|---:|---|
| h9_multi_head_h1_d8_seq8 | 1 | 8 | 8 | PASS |
| h9_multi_head_h2_d8_seq8 | 2 | 8 | 8 | PASS |
| h9_multi_head_h4_d8_seq8 | 4 | 8 | 8 | PASS |
| h9_multi_head_h2_d16_seq8 | 2 | 16 | 8 | PASS |
| h9_multi_head_h1_d64_seq8 | 1 | 64 | 8 | PASS |
| h9_sequence_cache_full_h1_d8_seq32 | 1 | 8 | 32 | PASS |

Coverage checked by the shared Stage 5 testbench:

- per-head output tile order;
- head index and K/V cache address correspondence;
- output metadata/status;
- `valid_seq_len` before and after every token;
- all-head completion before commit;
- cache-full extra token behavior;
- deterministic output/done backpressure;
- reset during provisional append;
- reset during attention start.

The H1/D64 run is a multi-head-compatible D64 check with one head. The tested
multi-head configurations are H2/D8, H4/D8, and H2/D16.

Limit: this is not the full requested reset or random-backpressure matrix.
