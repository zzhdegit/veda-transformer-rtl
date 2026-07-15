# ML-M3 Acceptance Audit

Status: **MODEL STAGE M3 IN PROGRESS - RTL/BIT-MODEL NUMERIC MISMATCH BLOCKED**

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
| Next-token RTL-assisted cases | BLOCKED | no complete RTL layer output |
| Hardware worktree modified | PASS | no model-line write to hardware repo |
| PDK/STA/P&R/PPA | PASS | not used |

ML-M3 cannot be accepted until the RTL/bit-model numerical mismatch is resolved in a separate hardware or reference-model fix task.
