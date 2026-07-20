# ML-M3 Cycle Comparison

| Length | H8 total | H9 total | Total delta H8-H9 | H8 attention | H9 attention | Attention delta H8-H9 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 127461 | 128285 | -824 | 1064 | 1888 | -824 |
| 2 | 255946 | 257346 | -1400 | 3152 | 4552 | -1400 |
| 8 | 1046344 | 1048008 | -1664 | 35168 | 36832 | -1664 |
| 16 | 2152176 | 2145680 | 6496 | 129824 | 123328 | 6496 |
| 32 | 4542016 | 4490016 | 52000 | 497312 | 445312 | 52000 |

Positive delta means the interleaved H9 schedule used fewer cycles than staged H8 for that field.
