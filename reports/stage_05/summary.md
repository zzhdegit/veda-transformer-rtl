# Stage 5 Summary

## Result
STAGE 5 PASS.

shared multi-head correctness accepted.

throughput and physical memory provisional.

## Scope
Stage 5 adds shared multi-head causal generation attention. It preserves the
Stage 4.1 current-token semantics and extends them across `N_HEAD` heads using
one shared Stage 3 single-head compute path.

Implemented:
- parameterized `N_HEAD`;
- head-banked token-major K/V cache;
- serial multi-head Q/K/V input;
- all-head provisional K/V transaction;
- shared single-head attention compute path;
- head 0 to `N_HEAD-1` sequential scheduling;
- per-head FP32 output stream;
- all-head atomic commit;
- Python bit model, VCS tests, assertions, vlogan lint, and DC elaboration.

Not implemented:
- QKV projection;
- output projection;
- head concat plus projection;
- FFN;
- LayerNorm/RMSNorm;
- full Transformer layer;
- eviction, voting, sliding window, P&R, or PPA.

## Semantics
For each generation token `t` and head `h`:

```text
output[h] = Attention(q[h,t], K[h,0:t], V[h,0:t])
```

Current K/V participates in every head's attention. All heads share one
committed `valid_seq_len`, and the current token commits only after all heads
finish successfully.

If any head fails before commit:
- no committed length increment occurs;
- all provisional head state is aborted;
- provisional memory contents may be overwritten by the next legal token.

If the cache is full:
- no K/V write occurs;
- no attention output is produced;
- `valid_seq_len` is unchanged;
- invalid status `8'h82` is reported.

## Implementation
- `rtl/cache/multi_head_kv_cache_manager.sv`
  - Stores K/V as `K_cache[head][token][dimension]`.
  - Uses linear address
    `((head * MAX_SEQ_LEN) + token) * D_HEAD + dimension`.
  - Tracks per-head provisional completion and one all-head commit.
- `rtl/cache/multi_head_generation_controller.sv`
  - Validates serial input order.
  - Writes all provisional K/V before starting any head.
  - Drives the shared single-head path with `start_seq_len = valid_seq_len + 1`.
  - Advances heads only after current head done.
  - Commits only after all heads done.
- `rtl/attention/multi_head_generation_engine.sv`
  - Instantiates one cache manager, one controller, and one shared
    `single_head_attention`.

## Coverage
Python tests cover:
- `N_HEAD` 1, 2, and 4;
- `D_HEAD` 8 and 16;
- sequence lengths 1, 2, 3, and 8/MAX;
- first-token self attention;
- second-token history+current attention;
- multi-step per-head bit-model equality;
- `N_HEAD=1` equality with Stage 4.1;
- head-banked addressing;
- provisional visibility and incomplete invisibility;
- cache full;
- reset/abort semantics through the reference model;
- atomic abort after a selected head failure.

RTL tests cover:
- `N_HEAD=1,D_HEAD=8`;
- `N_HEAD=2,D_HEAD=8`;
- `N_HEAD=4,D_HEAD=8`;
- `N_HEAD=2,D_HEAD=16`;
- cache provisional/commit/abort behavior;
- first token through full cache;
- output and done backpressure;
- token input stalls;
- reset during provisional append;
- reset during active attention;
- no output or commit when full.

## Assertions
Stage 5 includes checks for:
- `head_and_dim_order_legal`
- `no_head_start_before_all_provisional_complete`
- `each_head_attention_seq_len_equals_valid_plus_one`
- `no_next_head_before_current_done`
- `no_commit_before_all_heads_done`
- `no_partial_head_commit`
- `all_heads_share_valid_seq_len`
- `provisional_head_token_index_legal`
- `output_head_order_preserved`
- `no_overwrite_when_full`
- `abort_clears_all_head_provisional_state`
- `output stable under backpressure`
- `transaction count conserved`
- `no unknown output when valid`

## Verification
Host:

```bash
python scripts/sim/run_stage5_tests.py
python scripts/sim/run_stage4_tests.py
python scripts/sim/run_stage3_tests.py
```

PASS:
- Stage 5 model regression: 31 tests and py_compile.
- Stage 4.1 model regression: 23 tests and py_compile.
- Stage 3 model regression: 17 tests and py_compile.

Docker:

```bash
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage5-rtl-sim'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage5-lint'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage5-synth'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage4p1-rtl-sim'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage4p1-lint'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage4p1-synth'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage3-rtl-sim'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage3-lint'
docker exec nailong bash -lc 'cd /workspace/VEDA && make stage3-synth'
```

PASS:
- Stage 5 VCS: cache manager, H1/D8, H2/D8, H4/D8, H2/D16.
- Stage 5 vlogan lint: no diagnostics.
- Stage 5 DC analyze/elaborate/link/check_design.
- Stage 4.1 RTL/lint/DC post-change regression.
- Stage 3 RTL/lint/DC post-change regression.

Docker `make stage3-test` was attempted but fails due the container Python not
supporting existing repository files with `from __future__ import annotations`.
Host Python is used for model tests; Docker is used for VCS/vlogan/DC.

## RTL Counter Snapshot
Counters are cumulative from reset. Step 8 is the full-cache error case.

| Config | final valid step | total | per_head_attention | head_switch | provisional_write | commit | cache_read | cache_write | pe_stall | sfu_stall | output_stall | peak_seq |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H1 D8 | 7 | 2924 | 2778 | 0 | 64 | 8 | 576 | 64 | 736 | 300 | 2 | 8 |
| H2 D8 | 7 | 5835 | 5553 | 8 | 128 | 8 | 1152 | 128 | 1472 | 600 | 1 | 8 |
| H4 D8 | 7 | 11671 | 11119 | 24 | 256 | 8 | 2304 | 256 | 2944 | 1200 | 15 | 8 |
| H2 D16 | 7 | 10338 | 9802 | 8 | 256 | 8 | 2304 | 256 | 2944 | 600 | 10 | 8 |

These are RTL cycle counters only. No PPA is claimed.

## Notes
- Stage 5 started with a dirty worktree and missing
  `stage4p1-attention-accepted` tag.
- Stage 4.1 compatibility path remains intact.
- Old Stage 4.1 cycle data is not directly comparable to Stage 5 multi-head
  cycles because Stage 5 serializes multiple heads through one shared datapath.
