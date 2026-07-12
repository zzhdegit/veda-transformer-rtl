# Stage 4.1 Summary

## Result
STAGE 4.1 PASS.

standard causal self-attention current-token semantics accepted.

## What Changed
Stage 4 originally ran:

```text
attention(q_current, old_cache)
emit output
append current K/V
```

Stage 4.1 now runs:

```text
receive q_current/k_current/v_current
provisionally write current K/V at physical token valid_seq_len
attention(q_current, K_cache[0:current], V_cache[0:current])
emit output
commit token after attention done
```

The first token now starts Stage 3 with `seq_len=1` and produces output from
current `V_0`. The old fixed zero-output path was removed.

Old append-after-attention cycle data is not directly comparable with this
stage because every valid generation step now loads one additional K/V row and
the first token performs real attention.

## Implementation
- `kv_cache_manager` separates physical provisional write, current-operation
  visibility, and global commit.
- `generation_attention_controller` writes current K/V before cache load, starts
  Stage 3 with `old_valid_seq_len + 1`, and commits only after Stage 3 done.
- `generation_attention_engine` exposes the new provisional/commit controls and
  performance counters.
- Python reference models now use current K/V in the bit model before commit.
- Stage 4 vectors regenerate from the new reference model.

## Coverage
Tests cover:
- first token self-attention and nonzero output;
- second token attending both token 0 and token 1;
- multi-token steps using `t+1` K/V rows;
- provisional complete visibility and incomplete invisibility;
- no `valid_seq_len` increment before commit;
- abort/reset clearing provisional state;
- cache full with no write, output, overwrite, or length increment;
- token input stall, cache write stall, Stage 3 load/runtime stalls, output
  stall, and done stall;
- D_HEAD 8 and 16;
- generation steps 1, 2, 3, and 8/MAX_SEQ_LEN;
- Stage 3 regression preservation.

## Assertions
Static lint and VCS include checks for:
- `no_attention_without_complete_current_kv`
- `attention_seq_len_equals_valid_seq_len_plus_one`
- `provisional_token_index_equals_valid_seq_len`
- `no_commit_before_attention_done`
- `no_valid_seq_len_increment_before_commit`
- `no_provisional_visibility_before_complete`
- `current_token_read_allowed_only_for_active_operation`
- `no_overwrite_when_full`
- `abort_or_reset_clears_provisional_state`
- `first_token_not_empty_attention`

## Verification
Host:

```bash
python scripts/sim/run_stage4_tests.py
python scripts/sim/run_stage3_tests.py
```

PASS:
- Stage 4.1 model regression: 23 tests and py_compile.
- Stage 3 model regression: 17 tests and py_compile.
- Host VCS skipped because host has no `vcs`.

Docker:

```bash
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage4p1-test'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage4p1-lint'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage4p1-synth'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage3-rtl-sim'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage3-lint'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage3-synth'
```

PASS:
- Stage 4.1 Docker Python fallback: 23 tests.
- Stage 4.1 py_compile: PASS.
- Stage 4.1 VCS: `kv_cache_manager`, `generation_attention_d8`, and
  `generation_attention_d16` PASS.
- Stage 4.1 vlogan lint: PASS with no diagnostics.
- Stage 4.1 DC analyze/elaborate/link/check_design: PASS.
- Stage 3 Docker RTL/lint/DC regression: PASS.

DC checks are analyze/elaborate/link/check_design only. No PPA is claimed.

## RTL Counter Snapshot
Counters are cumulative from reset. Step 8 is the cache-full error case.

| D_HEAD | step | seq_before | seq_after | total | attention | provisional_append | commit | cache_read | cache_write | cache_stall | pe_stall | sfu_stall | peak_seq |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0 | 0 | 1 | 106 | 88 | 8 | 1 | 16 | 8 | 0 | 22 | 6 | 1 |
| 8 | 1 | 1 | 2 | 296 | 259 | 16 | 2 | 48 | 16 | 0 | 64 | 30 | 2 |
| 8 | 2 | 2 | 3 | 557 | 502 | 24 | 3 | 96 | 24 | 0 | 126 | 60 | 3 |
| 8 | 3 | 3 | 4 | 888 | 815 | 32 | 4 | 160 | 32 | 0 | 208 | 96 | 4 |
| 8 | 4 | 4 | 5 | 1290 | 1199 | 40 | 5 | 240 | 40 | 0 | 310 | 138 | 5 |
| 8 | 5 | 5 | 6 | 1763 | 1654 | 48 | 6 | 336 | 48 | 0 | 432 | 186 | 6 |
| 8 | 6 | 6 | 7 | 2307 | 2180 | 56 | 7 | 448 | 56 | 0 | 574 | 240 | 7 |
| 8 | 7 | 7 | 8 | 2924 | 2778 | 64 | 8 | 576 | 64 | 0 | 736 | 300 | 8 |
| 8 | 8 | 8 | 8 | 2934 | 2778 | 64 | 8 | 576 | 64 | 0 | 736 | 300 | 8 |
| 16 | 0 | 0 | 1 | 191 | 157 | 16 | 1 | 32 | 16 | 0 | 44 | 6 | 1 |
| 16 | 1 | 1 | 2 | 522 | 453 | 32 | 2 | 96 | 32 | 0 | 128 | 30 | 2 |
| 16 | 2 | 2 | 3 | 979 | 876 | 48 | 3 | 192 | 48 | 0 | 252 | 60 | 3 |
| 16 | 3 | 3 | 4 | 1564 | 1427 | 64 | 4 | 320 | 64 | 0 | 416 | 96 | 4 |
| 16 | 4 | 4 | 5 | 2275 | 2104 | 80 | 5 | 480 | 80 | 0 | 620 | 138 | 5 |
| 16 | 5 | 5 | 6 | 3114 | 2909 | 96 | 6 | 672 | 96 | 0 | 864 | 186 | 6 |
| 16 | 6 | 6 | 7 | 4079 | 3840 | 112 | 7 | 896 | 112 | 0 | 1148 | 240 | 7 |
| 16 | 7 | 7 | 8 | 5171 | 4898 | 128 | 8 | 1152 | 128 | 0 | 1472 | 300 | 8 |
| 16 | 8 | 8 | 8 | 5189 | 4898 | 128 | 8 | 1152 | 128 | 0 | 1472 | 300 | 8 |

## Limitations
- K/V cache remains behavioral memory only.
- No SRAM macro PPA, banking conclusion, timing closure, or layout result.
- No multi-head, projections, Transformer layer, eviction, voting, or sliding
  window.
- No Stage 5 work started.
