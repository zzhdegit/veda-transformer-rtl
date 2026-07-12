# Stage 4 Summary

## Result
STAGE 4 PASS.

Dynamic KV correctness accepted. Physical memory implementation provisional.
No PPA claim is made.

## Implemented
- Token-major dynamic K/V cache manager.
- Linear address generation: `address = token * D_HEAD + dimension`.
- Dimension-serial current-token `q/k/v` input.
- Continuous generation wrapper around Stage 3 `single_head_attention`.
- Append-after-attention semantics.
- Empty-cache zero-output path.
- Cache-full error path without overwrite.
- Stage 4 performance counters.
- Stage 4 Python reference models, VCS testbenches, lint, DC elaboration, and
  Makefile targets.

## Key Semantics
Each generation step uses the cache that was valid before the current token:

```text
attention(q_current, K_cache[0:valid_seq_len], V_cache[0:valid_seq_len])
append(k_current, v_current)
valid_seq_len += 1
```

The current token does not attend to itself. Empty cache returns FP32 zero output
tiles and then appends. Full cache returns `done_invalid=1`, `done_status=8'h82`,
and leaves `valid_seq_len` unchanged.

## Verification
Host:

```bash
python scripts/sim/run_stage4_tests.py
```

PASS: 23 model tests and py_compile.

Docker:

```bash
make stage4-test
make stage4-rtl-sim
make stage4-lint
make stage4-synth
```

PASS:
- `kv_cache_manager`
- `generation_attention_d8`
- `generation_attention_d16`
- vlogan lint with no diagnostics
- DC analyze/elaborate/link/check_design

DC checks include `generation_attention_engine D_HEAD=8/16`, `kv_cache_manager`
small configurations, and `kv_address_generator D_HEAD=128/MAX_SEQ_LEN=4096`.
The full D128/MAX4096 cache memory is not elaborated as flip-flop behavioral
storage because physical memory remains provisional.

Stage 3 regression remains PASS:

```bash
python scripts/sim/run_stage3_tests.py
make stage3-rtl-sim
make stage3-lint
make stage3-synth
```

## RTL Cycle Counters
Counters are cumulative from reset. Step 8 is the cache-full error case.

| D_HEAD | step | seq_before | seq_after | total | attention | append | cache_read | cache_write | cache_stall | pe_stall | sfu_stall | peak_seq |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0 | 0 | 1 | 19 | 2 | 8 | 0 | 8 | 0 | 0 | 0 | 1 |
| 8 | 1 | 1 | 2 | 125 | 90 | 16 | 16 | 16 | 0 | 22 | 6 | 2 |
| 8 | 2 | 2 | 3 | 313 | 261 | 24 | 48 | 24 | 0 | 64 | 30 | 3 |
| 8 | 3 | 3 | 4 | 573 | 503 | 32 | 96 | 32 | 0 | 126 | 60 | 4 |
| 8 | 4 | 4 | 5 | 903 | 816 | 40 | 160 | 40 | 0 | 208 | 96 | 5 |
| 8 | 5 | 5 | 6 | 1304 | 1200 | 48 | 240 | 48 | 0 | 310 | 138 | 6 |
| 8 | 6 | 6 | 7 | 1777 | 1656 | 56 | 336 | 56 | 0 | 432 | 186 | 7 |
| 8 | 7 | 7 | 8 | 2320 | 2182 | 64 | 448 | 64 | 0 | 574 | 240 | 8 |
| 8 | 8 | 8 | 8 | 2330 | 2182 | 64 | 448 | 64 | 0 | 574 | 240 | 8 |
| 16 | 0 | 0 | 1 | 37 | 3 | 16 | 0 | 16 | 0 | 0 | 0 | 1 |
| 16 | 1 | 1 | 2 | 228 | 160 | 32 | 32 | 32 | 0 | 44 | 6 | 2 |
| 16 | 2 | 2 | 3 | 557 | 456 | 48 | 96 | 48 | 0 | 128 | 30 | 3 |
| 16 | 3 | 3 | 4 | 1014 | 879 | 64 | 192 | 64 | 0 | 252 | 60 | 4 |
| 16 | 4 | 4 | 5 | 1598 | 1430 | 80 | 320 | 80 | 0 | 416 | 96 | 5 |
| 16 | 5 | 5 | 6 | 2308 | 2107 | 96 | 480 | 96 | 0 | 620 | 138 | 6 |
| 16 | 6 | 6 | 7 | 3147 | 2913 | 112 | 672 | 112 | 0 | 864 | 186 | 7 |
| 16 | 7 | 7 | 8 | 4111 | 3844 | 128 | 896 | 128 | 0 | 1148 | 240 | 8 |
| 16 | 8 | 8 | 8 | 4129 | 3844 | 128 | 896 | 128 | 0 | 1148 | 240 | 8 |

## Limitations
- K/V cache uses behavioral memory only.
- No physical SRAM macro, banking, or PPA result.
- No overlap between attention and append or next-token input.
- No eviction, voting, sliding window, multi-head, projections, full
  Transformer, P&R, or formal PPA.
