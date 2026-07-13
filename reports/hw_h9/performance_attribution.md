# Hardware Stage H9 Performance Attribution

Status: attribution clarified; full RTL ablation is not available in this
repository without a larger controller split.

## Compared Paths

Matched RTL A/B compares:

- A: `ATTENTION_PE_ARCH=PAPER_ARRAY`, `ATTENTION_SCHEDULE=STAGED`
- C: `ATTENTION_PE_ARCH=PAPER_ARRAY`, `ATTENTION_SCHEDULE=INTERLEAVED`

Both use the same `single_head_attention` top, D_HEAD, sequence length, input
pattern, K/V values, DesignWare wrappers, clock/reset, output/done ready
environment, and counter interval.

The useful architectural effects are:

1. H8 low-lane staged mapping.
2. H9 native full-array mapping.
3. H9 SFU/PE element-serial interleaving.

The current matched RTL A/B measures the combined H9 native mapping plus
interleaving result against the staged paper path. It does not isolate the
interleaving-only contribution.

## Non-interleaved Ablation

A native full-array non-interleaved RTL schedule would require a separate
paper-native controller that uses the H9 mapping while deliberately disabling
QK-SFU and SFU-sV overlap. That split is not present in the current RTL and
would be a new schedule path, not a small report-only switch.

Repository design decision: do not add a large ablation-only controller in
Hardware Stage H9 Final Closure. The Python model reports
`full_array_non_interleaved_cycles` as a trend estimate only.

## Performance Statement

The matched RTL performance gain must be stated as:

```text
H9 paper-native full-array mapping plus SFU/PE interleaving is faster than the
matched paper staged path at seq16 and seq32.
```

It must not be stated as:

```text
The full improvement is pure interleaving benefit.
```

Any 87% style number from earlier notes is an overall architecture comparison
number unless an RTL native full-array non-interleaved ablation is later added
and run.

## Matched RTL Summary

| D_HEAD | Seq | Staged RTL | Interleaved RTL | Improvement |
|---:|---:|---:|---:|---:|
| 8 | 16 | 1363 | 1169 | 14.23% |
| 8 | 32 | 2707 | 2209 | 18.40% |
| 16 | 16 | 2472 | 1171 | 52.63% |
| 16 | 32 | 4920 | 2211 | 55.06% |
| 64 | 16 | 9126 | 1183 | 87.04% |
| 64 | 32 | 18198 | 2223 | 87.78% |

The D64 numbers are dominated by eliminating the staged low-lane D_HEAD tiling
penalty and should not be attributed solely to overlap.
