# ML-M3 H8/H9 Real-Weight Comparison

| Case | H8 staged | H9 interleaved | H8 SHA | H9 SHA | H8/H9 captured output identical |
|---|---|---|---|---|---|
| length 1 partial | FAIL | FAIL | `5adbf7b5ef5e5fbff1a767e271d852ab711ec9d829a9f7fe9125288901d4f3be` | `5adbf7b5ef5e5fbff1a767e271d852ab711ec9d829a9f7fe9125288901d4f3be` | True |

Both schedules hit the same first mismatch against the hardware-aware bit model before completing token 0. This confirms the two accepted schedules agree on the captured prefix, but ML-M3 cannot claim bit-model equivalence.
