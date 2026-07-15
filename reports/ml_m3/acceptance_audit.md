# ML-M3 Acceptance Audit

Status: **MODEL STAGE M3 IN PROGRESS - HARDWARE NUMERIC FIX REQUIRED**

| Requirement | Result | Evidence |
|---|---|---|
| Q2 checkpoint SHA correct | PASS | `68b520f1322c79e568c39115809b8d623e21478af1662658cf997bf174cc9214` |
| Tokenizer SHA correct | PASS | `72c4100b9c923f8fc89ea563cdf18743742b87ad7cda6732606b61f50f290a1a` |
| 12 export tensors audited | PASS | tensor_count=12 |
| 8 RTL weight mappings | PASS | `reports/ml_m3/weight_mapping_audit.md` |
| H8/D8/D_MODEL64/D_FFN256 elaborate | PASS | VCS compile succeeded for staged and interleaved one-token smoke |
| H9 one-token real RTL | FAIL | CHECK_FAIL layer token=0 dim=1 got=3d4a2576 expected=3d4a2572 |
| H8 one-token real RTL | FAIL | same mismatch marker |
| length 1/2/8/16 real RTL | BLOCKED | one-token gate failed before multi-token |
| bit model vs H8/H9 | FAIL | first mismatch token 0 dim 1 |
| H8 vs H9 | PARTIAL_PASS | captured prefix SHA match=True |
| Full 64-dim diagnostic capture | PASS | 54/64 final dimensions mismatch bit model; H8/H9 captures identical |
| First divergent boundary | PASS | `w2_output_fp32_edge`; residual1 and norm2 edges match |
| First divergent arithmetic operation | PASS | W2 reduction-tree add at cycle 185551, tile base 8, width 8, pair 3 |
| Numeric replay | PASS | stable `fp32_add_wrapper` replay matches bit model/NumPy, not full PE path output |
| Root cause classification | PASS | common RTL PE reduction/stream-register numeric issue; hardware fix required |
| Next-token RTL-assisted cases | BLOCKED | no complete RTL layer output |
| Hardware worktree modified | PASS | no model-line write to hardware repo |
| PDK/STA/P&R/PPA | PASS | not used |

ML-M3 cannot be accepted until the common RTL PE reduction numeric issue is fixed on the hardware line and one-token H8/H9 bit-exact comparison passes against the hardware-aware bit model.
