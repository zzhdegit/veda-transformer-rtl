# ML-M3 Summary

Status: **MODEL STAGE M3 IN PROGRESS - HARDWARE NUMERIC FIX REQUIRED**

Completed:

- Q2 checkpoint/tokenizer/export/weight mapping audit passed.
- Real Q2 vectors were generated for lengths 1, 2, 8, 16, and 32.
- Reference-chain metrics were generated for PyTorch FP32, FP16-weight PyTorch, and the hardware-aware bit model.
- H8 staged and H9 interleaved transformer_layer both compiled/elaborated with N_HEAD=8, D_HEAD=8, D_MODEL=64, D_FFN=256, MAX_SEQ_LEN=128.
- H8 and H9 one-token diagnostic runs produced identical captured output.
- Full 64-dimension diagnostic capture found 54 final-output mismatches against the hardware-aware bit model.
- First stable divergent boundary was `w2_output_fp32_edge`, after matching `residual1_fp32_edge` and `norm2_output_fp16_edge`.
- First divergent arithmetic operation was W2 reduction-tree add at cycle 185551, tile base 8, width 8, pair 3: `3c81aa0c + 39699f40` produced `3c837d4b` in the full RTL PE path versus `3c837d4a` in the bit model.
- Standalone stable `fp32_add_wrapper` replay of the same operands matched the bit model and NumPy float32.

Blocked:

- First checked final-output boundary still mismatches bit model: `CHECK_FAIL layer token=0 dim=1 got=3d4a2576 expected=3d4a2572`.
- Root cause class: common RTL arithmetic path bug in the FFN W2 PE reduction/stream-register context.
- Multi-token RTL, full H8/H9 A/B, hybrid next-token logits, and M3 PASS are deferred until a hardware-owned fix closes the one-token bit-exact gate.

No hardware source, hardware report, PDK, STA, P&R, or PPA flow was modified or invoked.
