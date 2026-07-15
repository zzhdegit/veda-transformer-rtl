# ML-M3 Summary

Status: **MODEL STAGE M3 IN PROGRESS - RTL/BIT-MODEL NUMERIC MISMATCH BLOCKED**

Completed:

- Q2 checkpoint/tokenizer/export/weight mapping audit passed.
- Real Q2 vectors were generated for lengths 1, 2, 8, 16, and 32.
- Reference-chain metrics were generated for PyTorch FP32, FP16-weight PyTorch, and the hardware-aware bit model.
- H8 staged and H9 interleaved transformer_layer both compiled/elaborated with N_HEAD=8, D_HEAD=8, D_MODEL=64, D_FFN=256, MAX_SEQ_LEN=128.
- H8 and H9 one-token runs produced identical captured prefix output.

Blocked:

- First checked final-output boundary mismatched bit model: `CHECK_FAIL layer token=0 dim=1 got=3d4a2576 expected=3d4a2572`.
- Multi-token RTL, full H8/H9 A/B, hybrid next-token logits, and M3 PASS are deferred until the mismatch is fixed.

No hardware source, hardware report, PDK, STA, P&R, or PPA flow was modified or invoked.
