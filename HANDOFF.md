# Stage Handoff

## Stage
Stage 5: Shared Multi-Head Generation Attention

## Status
STAGE 5 PASS.

shared multi-head correctness accepted.

throughput and physical memory provisional.

## Completed
- Added a Stage 5 multi-head generation top that reuses one shared
  `single_head_attention` compute path.
- Added head-banked token-major K/V cache:
  - `rtl/cache/multi_head_kv_cache_manager.sv`
  - logical layout `K_cache[head][token][dimension]`
  - linear address `((head * MAX_SEQ_LEN) + token) * D_HEAD + dimension`
  - one shared committed `valid_seq_len`
- Added all-head provisional transaction semantics:
  - all heads write K/V provisional row `valid_seq_len`
  - current token is visible to the active operation only after all-head
    provisional completion
  - no committed `valid_seq_len` increment until all heads finish
  - abort clears all provisional head state
- Added `rtl/cache/multi_head_generation_controller.sv`.
  - Checks input order `head 0 dim 0..D_HEAD-1`, then next head.
  - Loads each head into the shared Stage 3 path.
  - Schedules heads strictly from 0 to `N_HEAD-1`.
  - Emits per-head output tiles and head/last metadata.
  - Commits once after all heads complete.
- Added `rtl/attention/multi_head_generation_engine.sv`.
  - Instantiates one cache manager, one controller, and one shared
    `single_head_attention`.
  - Does not instantiate N copies of `generation_attention_engine`.
- Added Python models:
  - `model/cache/multi_head_kv_cache_reference.py`
  - `model/cache/multi_head_generation_reference.py`
- Added Stage 5 tests and scripts:
  - `tb/model/test_stage5_multihead_generation.py`
  - `tb/rtl/stage5/tb_multi_head_kv_cache_manager.sv`
  - `tb/rtl/stage5/tb_multi_head_generation_engine.sv`
  - `scripts/sim/gen_stage5_vectors.py`
  - `scripts/sim/run_stage5_tests.py`
  - `scripts/sim/run_stage5_vcs.sh`
  - `scripts/lint/run_stage5_lint.py`
  - `scripts/synth/run_stage5_synth_check.py`
  - `scripts/synth/stage5_elaborate.tcl`
- Added Makefile targets:
  - `make stage5-test`
  - `make stage5-rtl-sim`
  - `make stage5-lint`
  - `make stage5-synth`
- Updated:
  - `PROJECT_STATE.md`
  - `reports/stage_05/summary.md`

## Not Completed
- QKV projection.
- Output projection.
- Head concat plus linear projection.
- FFN.
- LayerNorm/RMSNorm.
- Full Transformer layer.
- Voting, eviction, sliding window, or circular overwrite.
- Parallel multi-head compute engines.
- SRAM macro binding, P&R, STA, timing closure, power, area, frequency, or PPA.

## Start-of-Stage Notes
- Stage 5 branch is `stage5-shared-multihead`.
- Requested clean worktree was not present at stage start.
- Required tag `stage4p1-attention-accepted` was not present.
- Existing Stage 4/4.1 files and reports were already modified/untracked.
- No user or pre-existing changes were reverted or cleaned.
- Stage 4.1 baseline before Stage 5 edits passed host Python, Docker VCS,
  Docker vlogan lint, and Docker DC.

## Top Interface
`multi_head_generation_engine` is the Stage 5 top.

Token input is serial by head and dimension:
- `token_valid/token_ready`
- `token_head`
- `token_dim`
- `token_q_fp16`
- `token_k_fp16`
- `token_v_fp16`
- `token_last_dim`
- `token_last_head`
- `token_meta`

Required order:

```text
head 0 dim 0..D_HEAD-1
head 1 dim 0..D_HEAD-1
...
head N_HEAD-1 dim 0..D_HEAD-1
```

Output stream is per-head FP32 tiles:
- `output_valid/output_ready`
- `output_head`
- `output_base_dim`
- `output_vector_fp32`
- `output_lane_mask`
- `output_status`
- `output_invalid`
- `output_meta`
- `output_last_tile`
- `output_last_head`
- `output_last_token`

Done:
- `done_valid/done_ready`
- `done_status`
- `done_invalid`
- `done_meta`
- `done_valid_seq_len`
- `current_valid_seq_len`

Performance counters:
- `perf_generation_steps`
- `perf_total_cycles`
- `perf_per_head_attention_cycles`
- `perf_head_switch_cycles`
- `perf_provisional_write_cycles`
- `perf_cache_read_cycles`
- `perf_cache_write_cycles`
- `perf_cache_stall_cycles`
- `perf_commit_cycles`
- `perf_pe_stall_cycles`
- `perf_sfu_stall_cycles`
- `perf_output_stall_cycles`
- `perf_peak_valid_seq_len`

## Generation Semantics
For each generation token `t` and head `h`:

```text
output[h] = Attention(q[h,t], K[h,0:t], V[h,0:t])
```

The current token participates in every head's causal attention. The first token
therefore runs every head with `start_seq_len=1`.

All heads share one committed `valid_seq_len`. Commit is atomic across heads:

```text
receive all q/k/v heads
write all provisional K/V heads at token valid_seq_len
run head 0
run head 1
...
run head N_HEAD-1
emit every head output
commit current token once
valid_seq_len += 1
```

If the cache is full, Stage 5 does not write, output, overwrite, or commit. It
returns invalid status `8'h82` and leaves `valid_seq_len` unchanged.

If any head fails before commit, the controller aborts all provisional head
state and leaves committed `valid_seq_len` unchanged.

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

VCS runs use assertions through `-assert svaext`.

## Verification Results
Host:

```bash
python scripts/sim/run_stage5_tests.py
python scripts/sim/run_stage4_tests.py
python scripts/sim/run_stage3_tests.py
```

PASS:
- Stage 5 model tests and py_compile: 31 tests passed.
- Stage 4.1 model tests and py_compile: 23 tests passed.
- Stage 3 model tests and py_compile: 17 tests passed.
- Host RTL simulation skipped because host has no `vcs`.

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
- Stage 5 VCS:
  - `multi_head_kv_cache_manager`: PASS.
  - `N_HEAD=1,D_HEAD=8`: PASS.
  - `N_HEAD=2,D_HEAD=8`: PASS.
  - `N_HEAD=4,D_HEAD=8`: PASS.
  - `N_HEAD=2,D_HEAD=16`: PASS.
- Stage 5 vlogan lint: PASS with no diagnostics.
- Stage 5 DC analyze/elaborate/link/check_design: PASS.
- Stage 4.1 Docker VCS/lint/DC regression: PASS.
- Stage 3 Docker RTL/lint/DC regression: PASS.

Docker `make stage3-test` was attempted but fails because the container Python
does not support existing repository files using
`from __future__ import annotations`. This is a tooling/runtime limitation; host
Python Stage 3 tests pass and Docker Stage 3 VCS/lint/DC pass.

DC checks are analyze/elaborate/link/check_design only. No area, power, WNS,
frequency, process timing, or layout claim is produced.

## Dependencies
- Host Python for model tests.
- Docker container `nailong` for VCS, vlogan, and DC.
- Synopsys VCS/vlogan and Design Compiler inside the container.
- DesignWare simulation and foundation libraries inside the container.
- No PDK or standard-cell library path is required for the checked DC
  elaboration flow.

## Reproduction Steps
From `D:\IC_Workspace\VEDA`:

```bash
python scripts/sim/run_stage5_tests.py
python scripts/sim/run_stage4_tests.py
python scripts/sim/run_stage3_tests.py
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

## Next-Stage Cautions
- Do not start projection or Transformer layer work from this handoff.
- Preserve Stage 5 all-head atomic provisional/commit semantics.
- Preserve current-token causal attention for every head.
- Do not instantiate N copies of the full attention engine unless a future
  accepted spec explicitly changes the resource-sharing requirement.
- Do not make any provisional head globally committed before all heads complete.
- Cache full must not write, output, commit, or overwrite.
- Preserve Stage 3 ready/valid behavior and do not assume fixed PE/SFU latency.
- Keep behavioral cache memory out of PPA claims.

## Recommended Commit Message
```text
stage5: implement shared multi-head causal generation attention
```
