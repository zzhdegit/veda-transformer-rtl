# Hardware Stage H9 Matched RTL A/B Baseline

Status: PASS for matched single-head RTL baseline.

Command:

```text
docker exec -w /workspace/VEDA nailong make hw-h9-rtl-sim
```

Method:

- Same RTL top: `single_head_attention`.
- Same `ATTENTION_PE_ARCH=PAPER_ARRAY`.
- A: `ATTENTION_SCHEDULE=STAGED`.
- B: `ATTENTION_SCHEDULE=INTERLEAVED`.
- Same deterministic all-zero Q/K and all-one V inputs.
- Same existing Softmax arithmetic wrappers and DesignWare simulation library.
- Same clock, reset, output/done ready environment, and counter print path.
- No numerical tolerance was used; every valid output lane was checked as
  FP32 `1.0`.

The matched run also includes a deterministic output/done backpressure subset
for seq16 and seq32. It does not replace the still-open full random
backpressure matrix.

## No External Backpressure

### D_HEAD=8

| Seq | Paper staged RTL | Paper interleaved RTL | Delta | Improvement | QK-SFU overlap | SFU-sV overlap | Score peak | Prob peak | Group0 active | Group1 active |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 91 | 194 | -103 | -113.2% | 56 | 10 | 2 | 1 | 114 | 114 |
| 2 | 187 | 259 | -72 | -38.5% | 57 | 18 | 2 | 2 | 156 | 156 |
| 8 | 691 | 649 | 42 | 6.1% | 135 | 66 | 2 | 2 | 408 | 408 |
| 16 | 1363 | 1169 | 194 | 14.2% | 239 | 130 | 2 | 3 | 744 | 744 |
| 32 | 2707 | 2209 | 498 | 18.4% | 447 | 258 | 2 | 5 | 1416 | 1416 |
| 64 | 5395 | 4289 | 1106 | 20.5% | 863 | 514 | 2 | 8 | 2760 | 2760 |

### D_HEAD=16

| Seq | Paper staged RTL | Paper interleaved RTL | Delta | Improvement | QK-SFU overlap | SFU-sV overlap | Score peak | Prob peak | Group0 active | Group1 active |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 165 | 196 | -31 | -18.8% | 56 | 10 | 2 | 1 | 114 | 114 |
| 2 | 330 | 261 | 69 | 20.9% | 57 | 18 | 2 | 2 | 156 | 156 |
| 8 | 1248 | 651 | 597 | 47.8% | 135 | 66 | 2 | 2 | 408 | 408 |
| 16 | 2472 | 1171 | 1301 | 52.6% | 239 | 130 | 2 | 3 | 744 | 744 |
| 32 | 4920 | 2211 | 2709 | 55.1% | 447 | 258 | 2 | 5 | 1416 | 1416 |
| 64 | 9816 | 4291 | 5525 | 56.3% | 863 | 514 | 2 | 8 | 2760 | 2760 |

### D_HEAD=64

| Seq | Paper staged RTL | Paper interleaved RTL | Delta | Improvement | QK-SFU overlap | SFU-sV overlap | Score peak | Prob peak | Group0 active | Group1 active |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 609 | 208 | 401 | 65.8% | 56 | 10 | 2 | 1 | 114 | 114 |
| 2 | 1188 | 273 | 915 | 77.0% | 57 | 18 | 2 | 2 | 156 | 156 |
| 8 | 4590 | 663 | 3927 | 85.6% | 135 | 66 | 2 | 2 | 408 | 408 |
| 16 | 9126 | 1183 | 7943 | 87.0% | 239 | 130 | 2 | 3 | 744 | 744 |
| 32 | 18198 | 2223 | 15975 | 87.8% | 447 | 258 | 2 | 5 | 1416 | 1416 |
| 64 | 36342 | 4303 | 32039 | 88.2% | 863 | 514 | 2 | 8 | 2760 | 2760 |

## Deterministic Backpressure Subset

| D_HEAD | Seq | Paper staged RTL | Paper interleaved RTL | Delta | Staged output stalls | H9 output stalls |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 1363 | 1170 | 193 | 0 | 1 |
| 8 | 32 | 2708 | 2209 | 499 | 1 | 0 |
| 16 | 16 | 2472 | 1172 | 1300 | 0 | 1 |
| 16 | 32 | 4920 | 2213 | 2707 | 0 | 2 |
| 64 | 16 | 9131 | 1187 | 7944 | 5 | 4 |
| 64 | 32 | 18200 | 2225 | 15975 | 2 | 2 |

## Conclusion

The Hardware Stage H9 matched single-head RTL performance objective is met:
paper interleaved RTL is faster than paper staged RTL at seq16 and seq32 for
D_HEAD=8, 16, and 64 under no external backpressure and under the deterministic
output/done backpressure subset.

This result supersedes the earlier structural Python-cycle comparison for
performance acceptance. The earlier comparison was not apples-to-apples because
it compared an abstract cycle model against neither the staged nor interleaved
RTL with the same top, latencies, buffers, and ready/valid environment.

Hardware Stage H9 is still not accepted. Multi-head, full-layer,
long-sequence, cache-full, assertion bind/negative evidence, direct reset
matrix, and direct 20-seed random backpressure coverage have since executed and
passed in the final-closure run. The remaining blockers are strict independent
multi-head/full-layer reset injection coverage and broad multi-endpoint
multi-head/full-layer random-backpressure coverage.
