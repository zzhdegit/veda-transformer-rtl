# Hardware Stage H9 Multi-head Results

Status: RTL coverage entry implemented; not executed in the current environment.

Implemented entry:

```text
make hw-h9-multi-head-test
bash scripts/sim/run_hw_h9_vcs.sh
```

The H9 RTL script now compiles the existing bit-exact Stage 5
`multi_head_generation_engine` testbench with:

```text
ATTENTION_PE_ARCH=PAPER_ARRAY
ATTENTION_SCHEDULE=INTERLEAVED
```

Configured runs:

| Run | N_HEAD | D_HEAD | MAX_SEQ_LEN | Coverage |
|---|---:|---:|---:|---|
| h9_multi_head_h1_d8_seq8 | 1 | 8 | 8 | one token, two tokens, seq8, cache-full extra token |
| h9_multi_head_h2_d8_seq8 | 2 | 8 | 8 | head boundary, all-head completion, atomic commit |
| h9_multi_head_h4_d8_seq8 | 4 | 8 | 8 | multi-head state isolation |
| h9_multi_head_h2_d16_seq8 | 2 | 16 | 8 | D_HEAD tiling and two-head cache indexing |
| h9_multi_head_h1_d64_seq8 | 1 | 64 | 8 | D64 compatibility check |
| h9_sequence_cache_full_h1_d8_seq32 | 1 | 8 | 32 | long sequence and cache-full extra token |

The testbench checks output tiles against the Stage 5 Python reference, done
metadata/status, current `valid_seq_len`, output backpressure, reset during
provisional append, and reset during attention.

Current execution result:

```text
vcs: NOT FOUND
result=FAIL
```

Because VCS is unavailable in this environment, these new multi-head runs are
not counted as passing HW-H9 acceptance.
