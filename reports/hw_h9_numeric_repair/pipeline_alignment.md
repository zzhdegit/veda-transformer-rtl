# HW-H9-N1 Pipeline Alignment

The failing transaction was traced through FFN W2 lane product, reduction, tile
sum, tile accumulation, and final residual2 commit. Operand and metadata
alignment was checked against the real PE/reduction path.

## Alignment Table

| Signal | Source cycle relation | Register stages | Consume cycle relation |
|---|---:|---:|---:|
| Lane products | W2 tile launch | PE lane MAC latency plus core staging | Reduction input capture |
| L1 pair operands | Reduction `ST_ADD_START` | 0 before wrapper input fire | DW add combinational input |
| Add result payload | Wrapper input fire | 1 stream register stage | Reduction `ST_ADD_WAIT` result fire |
| Pair id | Reduction `pair_q` at launch | tracked in assertion shadow register | same `pair_q` at result fire |
| Active width | Reduction `width_q` at launch | tracked in assertion shadow register | same `width_q` at result fire |
| Metadata/last | Reduction input capture | wrapper stream register | final reduction output |
| Tile result | Reduction width 1 | PE core output stream register | tile accumulator write |
| W2 output commit | tile accumulator complete | transformer output stream | residual2 final add |

## Finding

No transaction misassociation was found for the first failing operation.
The actual DW add inputs for the W2 pair were:

```text
A = 32'h3c81aa0c
B = 32'h39699f40
```

The wrong result was produced from those current operands under the old add
wrapper rounding-mode encoding. It was not a previous-pair result, next-pair
result, stale pipeline result, tile-switch residue, invalid lane mask, repeated
partial sum, or omitted partial sum.
