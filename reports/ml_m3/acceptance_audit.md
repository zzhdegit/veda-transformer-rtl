# ML-M3 Acceptance Audit

Status: **MODEL STAGE M3 PASS**

| Requirement | Result | Evidence |
|---|---|---|
| repair tag audit | PASS | `hw-h9-real-weight-numeric-repair-accepted` -> `a54e608a8dc7e63c7e5dd342f8b893bb1e0b7485` |
| Docker actual hardware commit | PASS | `a54e608a8dc7e63c7e5dd342f8b893bb1e0b7485` |
| checkpoint SHA | PASS | `68b520f1322c79e568c39115809b8d623e21478af1662658cf997bf174cc9214` |
| tokenizer SHA | PASS | `72c4100b9c923f8fc89ea563cdf18743742b87ad7cda6732606b61f50f290a1a` |
| weight mapping | PASS | 8 RTL layer tensors |
| length1/2/8/16 H8/H9/bit-model | PASS | no-stall and output+done stall |
| internal node comparison | PASS | 9 categories per schedule |
| software full/incremental | PASS | max_abs=3.618001937866211e-05 |
| hybrid next-token | PASS | len1/8/16 top-1 agreement |
| continuous two-step | PASS | {'prompt0_token_ids': [1], 'prompt1_token_ids': [1, 230], 'prompt1_uses_step0_prediction': True, 'result': 'PASS', 'step0_bit_top1': 230, 'step0_h8_top1': 230, 'step0_h9_top1': 230, 'step1_bit_top1': 5, 'step1_h8_top1': 5, 'step1_h9_top1': 5} |
| forbidden-path audit | PASS | PDK/STA/P&R/PPA not used |
| length32 extended | PASS | no-stall H8/H9 |
