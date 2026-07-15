# ML-M3 Numeric Alignment Regression

| Check | Result |
|---|---|
| Reproduced one-token H8/H9 mismatch | PASS - mismatch reproduced |
| Diagnostic full 64-dim capture | PASS - 54 mismatches collected |
| H8/H9 identity | PASS - captures identical |
| Numeric replay | PASS - stable wrapper replay matches bit model |
| One-token bit-exact closure | FAIL - hardware numeric fix required |
| length2/8/16 continuation | NOT RUN by task boundary |
