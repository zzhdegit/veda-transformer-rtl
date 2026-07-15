# ML-M3 H8/H9 Real-Weight Comparison

| Length | H8 staged result | H9 interleaved result | H8 capture SHA | H9 capture SHA | Output identical |
|---:|---|---|---|---|---|
| 1 | DIAGNOSTIC | DIAGNOSTIC | `d09c4b80953c18ba7edea335490cca63f5ac4d9fcf24f4e17027813ca0045fb4` | `d09c4b80953c18ba7edea335490cca63f5ac4d9fcf24f4e17027813ca0045fb4` | True |

H8 staged and H9 interleaved remain numerically identical for the one-token
diagnostic capture. Both schedules differ from the hardware-aware bit model in
the same way, so the current blocker is a common arithmetic path issue and not
an interleaving-only scheduler issue.

The H8/H9 A/B cycle comparison for length 2/8/16 remains blocked by the
one-token bit-exact gate.
