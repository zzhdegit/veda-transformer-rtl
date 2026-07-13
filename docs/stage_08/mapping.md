# Stage 8 Attention Mapping

Stage 8 maps only the Attention QK and sV compute paths to the
Paper-Structured 8x8x2 PE Array. Projection WQ/WK/WV/WO and FFN W1/W2 remain
on the legacy `reconfigurable_pe_core` path.

## Architecture Select

The Attention PE architecture is selected by:

```text
ATTENTION_PE_ARCH = 0  legacy reconfigurable_pe_core
ATTENTION_PE_ARCH = 1  paper_array_8x8x2 through paper_attention_adapter
```

The default is legacy. The paper path is elaborated only when the parameter is
set to `1`.

## QK Mapping

QK uses `MODE_INNER_PRODUCT`.

| Item | Mapping |
|---|---|
| Q operand | current query tile, FP16 |
| K operand | token-major K cache row, FP16 |
| Permanent K transpose | none |
| Active dimensions | low `PE_NUM` lanes of the 8x8x2 array per legacy tile |
| Sequence tail | Stage 5 valid sequence controls which K tokens are issued |
| D_HEAD tail | lane mask from the legacy controller |
| Output | one FP32 score on the final D_HEAD tile for each token |
| Scale | unchanged existing `attention_score_scaler` after raw QK |

The repository implementation uses a minimal PE-like adapter so the existing
Stage 5 controller sequence remains unchanged. For the current Stage 5
`PE_NUM=8` configurations this maps each legacy tile into the first active
paper-array lanes and masks the rest. This preserves the Stage 5 bit pattern
for D8 and D16 tests while exposing the real 128-cell paper-array hierarchy.

## sV Mapping

sV uses `MODE_OUTER_PRODUCT`.

| Item | Mapping |
|---|---|
| Scalar | existing FP32 softmax probability |
| V operand | token-major V cache row, FP16 |
| Probability/V index | the existing Stage 5 controller issues matching indices |
| Partial sum | paper-array local accumulators within the active command |
| Output | FP32 vector tile on the final sequence update for the tile |
| D_HEAD tail | lane mask from the legacy controller |
| Sequence tail | existing valid sequence length and cache-full semantics |

Softmax remains staged serial:

```text
QK complete -> existing Softmax/SFU -> sV
```

Stage 8 does not implement QK/SFU/sV overlap.

## KV Cache Semantics

The mapping keeps the Stage 5 cache layout:

```text
K_cache[head][token][dimension]
V_cache[head][token][dimension]
```

The current-token provisional write, current-token participation,
all-head atomic commit, pre-commit error behavior, post-commit behavior,
cache-full behavior, reset behavior, and next-token-after-done rule are not
changed by Stage 8.

## Counters

The paper path exposes these cycle counters through single-head, multi-head,
projection-integrated MHA, and transformer-layer wrappers:

- `perf_paper_array_active_cycles`
- `perf_paper_array_idle_cycles`
- `perf_inner_mode_cycles`
- `perf_outer_mode_cycles`
- `perf_group0_active_cycles`
- `perf_group1_active_cycles`
- `perf_tail_masked_pe_cycles`
- `perf_mode_switch_cycles`
- `perf_array_input_stall_cycles`
- `perf_array_output_stall_cycles`

These are RTL cycle counters only. They are not timing, frequency, area, power,
or PPA conclusions.
