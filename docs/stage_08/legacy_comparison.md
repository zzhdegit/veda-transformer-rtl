# Stage 8 Legacy Comparison

Stage 8 preserves both Attention architectures:

| Architecture | QK/sV PE | Projection PE | FFN PE |
|---|---|---|---|
| `LEGACY_PE` | `reconfigurable_pe_core` | `reconfigurable_pe_core` path | `reconfigurable_pe_core` |
| `PAPER_ARRAY` | `paper_array_8x8x2` through adapter | unchanged legacy path | unchanged legacy path |

## Numeric Comparison

The paper-array bit model routes QK and sV through the explicit 128-cell
array model. The legacy model uses the frozen Stage 5 PE model.

For the Stage 8 D8/D16 coverage currently used by RTL regressions:

| Compared item | Result |
|---|---|
| QK raw score | bit-exact |
| Scaled score | bit-exact |
| Softmax probabilities | bit-exact |
| sV output | bit-exact |
| Final single-head output | bit-exact |
| Multi-head generation output | bit-exact |
| Full transformer-layer output | bit-exact for covered H1/D8, H2/D8, H4/D8, H2/D16, and H2/D8 two-token vectors |

The current adapter maps each legacy `PE_NUM=8` attention tile into masked
paper-array lanes. Because the effective FP32 add order matches the legacy
tile order in the covered D8/D16 cases, the paper path and legacy path are
bit-exact. Future wider or more parallel mappings may change add order; such
changes must report max absolute error, MAE, RMSE, relative L2, cosine
similarity, ULP difference, and Attention ranking consistency.

## Structural Comparison

DC hierarchy reports for Stage 8D counted 128 `paper_pe_cell` occurrences in:

- `single_head_attention` with `ATTENTION_PE_ARCH=1`
- `multi_head_generation_engine` with `ATTENTION_PE_ARCH=1`
- `transformer_layer` with `ATTENTION_PE_ARCH=1`

Legacy architecture elaboration remains available and is checked separately.

## Limitations

- The paper path is a correctness-first adapter, not a throughput-optimized
  global array schedule.
- Tail waste is high for current `PE_NUM=8` Stage 5 tiles because most of the
  128 physical cells are masked.
- Softmax remains serial and staged.
- Projection and FFN are not migrated to the paper array in Stage 8.
- DC reports are structural only and are not PPA results.
