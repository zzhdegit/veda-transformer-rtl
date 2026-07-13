# Hardware Stage H9 Full-Array Mapping

H8 mapped the existing `PE_NUM=8` legacy tile into low paper-array lanes. H9 adds a paper-native mapping for the interleaved paper path.

## Dimension Mapping

For `D_HEAD <= 128`:

```text
group  = dimension % 2
local  = dimension / 2
row    = local % 8
column = local / 8
```

The physical cell index is:

```text
group * 64 + row * 8 + column
```

Consequences:

- D_HEAD=8 uses both groups and multiple rows.
- D_HEAD=16 uses both groups and all rows.
- D_HEAD=64 uses both groups, all rows, and four columns per group.
- D_HEAD=128 uses all 128 PE cells.

## QK

Q is broadcast logically by placing each Q dimension at its native cell. K remains token-major. Each transaction produces one score for one K token and one head.

L1 reduces columns within each row. L2 reduces rows within each group. The two group sums are then combined.

## sV

One probability scalar is broadcast to all active native cells. V remains token-major. Each probability updates each active output dimension exactly once. The output is gathered back into logical dimension order.

## Tails

Sequence tails are handled by score/probability packet count. Dimension tails are handled by the native lane mask. Inactive cells do not accumulate.
