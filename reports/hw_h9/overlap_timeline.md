# Hardware Stage H9 Overlap Timeline

Example structural timeline for D_HEAD=64, seq_len=8:

| Cycle | Array mode | Array work | SFU work | Score FIFO | Prob FIFO |
| ----: | ---------- | ---------- | -------- | ---------: | --------: |
| 0 | INNER | score 0 issue | idle | 0 | 0 |
| 10 | INNER | score 1 issue | reduce score 0 | 0 | 0 |
| 20 | INNER | score 2 issue | reduce score 1 | 0 | 0 |
| 30 | INNER | score 3 issue | reduce score 2 | 0 | 0 |
| 40 | INNER | score 4 issue | reduce score 3 | 0 | 0 |
| 50 | INNER | score 5 issue | reduce score 4 | 0 | 0 |
| 60 | INNER | score 6 issue | reduce score 5 | 0 | 0 |
| 70 | INNER | score 7 issue | reduce score 6 | 0 | 0 |
| 82 | DRAIN | inner drained | final softmax state | 0 | 0 |
| 83 | SWITCH | inner to outer | recip/setup | 0 | 0 |
| 91 | OUTER | consume prob 0 | normalize prob 1 | 0 | 0 |
| 99 | OUTER | consume prob 1 | normalize prob 2 | 0 | 0 |
| 107 | OUTER | consume prob 2 | normalize prob 3 | 0 | 0 |

This table demonstrates allowed overlap only:

- QK PE with SFU reduction;
- SFU normalization with sV PE.

It does not claim simultaneous QK and sV use of the same array.
