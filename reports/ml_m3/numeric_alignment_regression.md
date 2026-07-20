# ML-M3 Numeric Alignment Regression

## Closure Note

This table is retained as a historical pre-repair regression. The repaired baseline `hw-h9-real-weight-numeric-repair-accepted` (`a54e608a8dc7e63c7e5dd342f8b893bb1e0b7485`) closes one-token and length2/8/16 bit-exact RTL co-simulation with zero mismatches. Current regression status is recorded in `reports/ml_m3/regression.md` and `reports/ml_m3/acceptance_audit.md`.

## Historical Regression

| Check | Result |
|---|---|
| Reproduced one-token H8/H9 mismatch | PASS - mismatch reproduced |
| Diagnostic full 64-dim capture | PASS - 54 mismatches collected |
| H8/H9 identity | PASS - captures identical |
| Numeric replay | PASS - stable wrapper replay matches bit model |
| One-token bit-exact closure | FAIL - hardware numeric fix required |
| length2/8/16 continuation | NOT RUN by task boundary |
