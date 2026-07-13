# Hardware Stage H9 Specification

Hardware Stage H9 (HW-H9) implements paper Attention improvements only:

1. Full-array paper Attention mapping.
2. SFU-PE element-serial interleaving for the paper Attention path.

Projection WQ/WK/WV/WO and FFN W1/W2 remain on the existing paths.

## Architecture Select

Existing architecture select remains:

```text
ATTENTION_PE_ARCH = 0  LEGACY_PE
ATTENTION_PE_ARCH = 1  PAPER_ARRAY
```

H9 adds a schedule select:

```text
ATTENTION_SCHEDULE = 0  STAGED
ATTENTION_SCHEDULE = 1  INTERLEAVED
```

Legal combinations:

- LEGACY_PE + STAGED
- PAPER_ARRAY + STAGED
- PAPER_ARRAY + INTERLEAVED

LEGACY_PE + INTERLEAVED is illegal.

## Numeric Contract

H9 preserves:

- FP16 operands.
- Exact FP16-to-FP32 expansion.
- FP32 products, partial sums, accumulators, and score scale.
- Current Stage 3 softmax arithmetic.
- Current FP32-to-FP16 conversion boundaries.
- Current-token causal semantics.
- K/V layout: `K_cache[head][token][dimension]`, `V_cache[head][token][dimension]`.
- All-head atomic commit and cache-full behavior.

H9 native mapping changes FP32 add order relative to H8 for some cases. H9 RTL must match the H9 bit model. H8 differences are reported with error metrics and are not hidden.

## Full-Array Mapping

For one head tile up to 128 dimensions:

```text
group  = dimension % 2
local  = dimension / 2
row    = local % 8
column = local / 8
```

This maps small heads across rows and both groups instead of only the lowest contiguous PE cells. D_HEAD greater than 128 remains out of scope for HW-H9.

## Interleaved Schedule

The paper-array interleaved schedule is:

```text
INNER issue score packets
|| SFU online reduction over prior score packets
-> inner drain
-> final softmax state valid
-> mode switch
-> SFU normalization emits probability packets
|| OUTER consumes prior probability packets for sV
-> outer drain
-> retire
```

One paper array cannot issue INNER and OUTER commands in the same cycle.

## Exit Boundary

H9 does not claim paper-exact RTL, SRAM macro binding, eviction, global array sharing, Projection/FFN migration, STA, P&R, area, power, frequency, WNS, or PPA.
