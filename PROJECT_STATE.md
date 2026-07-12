# Project State

## Current Stage
- Stage: 5
- Status: STAGE 5 PASS
- Branch observed after setup: `stage5-shared-multihead`
- Last update: 2026-07-12

## Stage 5 Result
shared multi-head correctness accepted.

throughput and physical memory provisional.

Stage 5 implements shared multi-head causal generation attention on top of the
accepted Stage 4.1 current-token semantics. For each generation token, all heads
receive complete Q/K/V, all heads write provisional K/V for the current token,
one shared `single_head_attention` compute path runs head 0 through
`N_HEAD-1` sequentially, and the token commits only after every head output and
done complete successfully.

## Start-of-Stage Notes
- Stage 5 requested branch was created as `stage5-shared-multihead`.
- The requested clean starting worktree was not present. Stage 4/4.1 files and
  generated reports were already modified or untracked.
- Required tag `stage4p1-attention-accepted` was not present.
- No existing workspace changes were reverted, cleaned, or overwritten.
- Pre-change Stage 4.1 baseline passed:
  - Host `python scripts/sim/run_stage4_tests.py`: PASS, 23 tests.
  - Docker `make stage4p1-rtl-sim`: PASS.
  - Docker `make stage4p1-lint`: PASS.
  - Docker `make stage4p1-synth`: PASS.

## Frozen Decisions
- Stage 4.1 standard causal current-token semantics are preserved.
- Logical K/V layout is head-banked token-major:

```text
K_cache[head][token][dimension]
V_cache[head][token][dimension]
address = ((head * MAX_SEQ_LEN) + token) * D_HEAD + dimension
```

- All heads share one committed `valid_seq_len`.
- Current K/V is provisional for all heads until the full multi-head token
  transaction commits.
- One shared Stage 3 single-head compute path is time-multiplexed by head.
- No QKV projection, output projection, concat projection, FFN, LayerNorm,
  Transformer layer, eviction, voting, sliding window, P&R, or PPA is
  implemented.
- Behavioral cache memory remains provisional and is not an SRAM macro or PPA
  basis.
- DesignWare remains confined to existing project wrappers.

## Stage 5 Generation Semantics
For generation step `t` with old committed length `valid_seq_len`:

1. Receive `q[h]`, `k[h]`, and `v[h]` for every head in head/dimension order.
2. If `valid_seq_len == MAX_SEQ_LEN`, return `done_invalid=1`, status `8'h82`,
   no output, no write, and no length increment.
3. Write every head's K/V to provisional row `valid_seq_len`.
4. Keep committed `valid_seq_len` unchanged while provisional data is in flight.
5. For each head `h = 0..N_HEAD-1`, load that head's q and K/V rows
   `0..valid_seq_len` into the shared single-head attention path.
6. Start each head with `start_seq_len = old_valid_seq_len + 1`.
7. Forward per-head FP32 output tiles in head order.
8. Commit the token only after all heads complete successfully.
9. If any head fails before commit, abort all provisional head state and leave
   `valid_seq_len` unchanged.

Mathematically:

```text
output[h] = Attention(q[h,t], K[h,0:t], V[h,0:t])
```

where `K[h,t]/V[h,t]` are the current token's provisional K/V for head `h`.

## RTL Changes
- Added `rtl/cache/multi_head_kv_cache_manager.sv`
  - Head-banked token-major K/V memory.
  - Per-head provisional completion with all-head atomic commit.
  - One shared committed `valid_seq_len`.
  - Abort clears all provisional head state without changing committed length.
- Added `rtl/cache/multi_head_generation_controller.sv`
  - Validates strict head/dimension token input order.
  - Writes all heads' provisional K/V before any head starts attention.
  - Schedules the shared single-head datapath from head 0 to `N_HEAD-1`.
  - Emits per-head output metadata and commits only after all heads complete.
- Added `rtl/attention/multi_head_generation_engine.sv`
  - Instantiates one `multi_head_kv_cache_manager`.
  - Instantiates one shared `single_head_attention`.
  - Instantiates one `multi_head_generation_controller`.
  - Does not instantiate `generation_attention_engine` or N copies of Stage 3.

Stage 4.1 `generation_attention_engine` remains available as the single-head
compatibility path.

## Model And Test Changes
- Added `model/cache/multi_head_kv_cache_reference.py`.
- Added `model/cache/multi_head_generation_reference.py`.
- Added `tb/model/test_stage5_multihead_generation.py`.
- Added Stage 5 vector generation and regressions:
  - `scripts/sim/gen_stage5_vectors.py`
  - `scripts/sim/run_stage5_tests.py`
  - `scripts/sim/run_stage5_vcs.sh`
  - `scripts/lint/run_stage5_lint.py`
  - `scripts/synth/run_stage5_synth_check.py`
  - `scripts/synth/stage5_elaborate.tcl`
- Added RTL testbenches:
  - `tb/rtl/stage5/tb_multi_head_kv_cache_manager.sv`
  - `tb/rtl/stage5/tb_multi_head_generation_engine.sv`
- Added Makefile targets:
  - `make stage5-test`
  - `make stage5-rtl-sim`
  - `make stage5-lint`
  - `make stage5-synth`

## Assertions
Stage 5 RTL and lint include assertion/stability tokens for:
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

## Verification Performed
Host Python:

```bash
python scripts/sim/run_stage5_tests.py
python scripts/sim/run_stage4_tests.py
python scripts/sim/run_stage3_tests.py
```

PASS:
- Stage 5 model regression: 31 tests and py_compile.
- Stage 4.1 model regression: 23 tests and py_compile.
- Stage 3 model regression: 17 tests and py_compile.
- Host RTL simulation skipped by scripts because host has no `vcs`.

Docker VCS/lint/DC:

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
- Stage 5 VCS:
  - `multi_head_kv_cache_manager`: PASS.
  - `N_HEAD=1,D_HEAD=8`: PASS.
  - `N_HEAD=2,D_HEAD=8`: PASS.
  - `N_HEAD=4,D_HEAD=8`: PASS.
  - `N_HEAD=2,D_HEAD=16`: PASS.
- Stage 5 vlogan lint: PASS with no diagnostics.
- Stage 5 DC analyze/elaborate/link/check_design: PASS.
- Stage 4.1 VCS/lint/DC post-change regression: PASS.
- Stage 3 RTL/lint/DC post-change regression: PASS.

Note: Docker `make stage3-test` was also tried, but that target uses the
container's old Python, which does not support existing repository files with
`from __future__ import annotations`. Host Python is the Stage 3/4/5 model-test
source of truth for this run; Docker remains the VCS/vlogan/DC environment.

DC checks are analyze/elaborate/link/check_design only. They do not generate
area, power, WNS, frequency, layout, or process timing claims.

## Stage 5 RTL Counter Snapshot
Counters are cumulative from reset. Step 8 is the full-cache error case.

| Config | final valid step | total | per_head_attention | head_switch | provisional_write | commit | cache_read | cache_write | pe_stall | sfu_stall | output_stall | peak_seq |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H1 D8 | 7 | 2924 | 2778 | 0 | 64 | 8 | 576 | 64 | 736 | 300 | 2 | 8 |
| H2 D8 | 7 | 5835 | 5553 | 8 | 128 | 8 | 1152 | 128 | 1472 | 600 | 1 | 8 |
| H4 D8 | 7 | 11671 | 11119 | 24 | 256 | 8 | 2304 | 256 | 2944 | 1200 | 15 | 8 |
| H2 D16 | 7 | 10338 | 9802 | 8 | 256 | 8 | 2304 | 256 | 2944 | 600 | 10 | 8 |

These are RTL cycle counters only. They are not PPA.

## Known Limitations
- The multi-head schedule is intentionally serial by head.
- K/V cache memory is still behavioral and not physically banked SRAM.
- No projection, concat, output projection, Transformer layer, eviction,
  voting, PPA, timing closure, or layout is included.

## Next Action
- Do not start projection or Transformer layer work from this stage.
- A future projection stage must preserve Stage 5's all-head atomic
  provisional/commit behavior unless a later accepted spec changes RTL, models,
  tests, and documentation together.
