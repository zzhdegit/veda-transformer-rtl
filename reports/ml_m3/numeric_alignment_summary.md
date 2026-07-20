# ML-M3 Numeric Alignment Summary

Status: **MODEL STAGE M3 PASS** on repaired hardware baseline `hw-h9-real-weight-numeric-repair-accepted` (`a54e608a8dc7e63c7e5dd342f8b893bb1e0b7485`).

The historical one-token dim1 mismatch is closed. Length1/2/8/16 H8 staged RTL, H9 interleaved RTL, and the hardware-aware bit model are bit-exact with zero mismatches. See `reports/ml_m3/acceptance_audit.md`, `reports/ml_m3/rtl_smoke_results.md`, and `reports/ml_m3/node_comparison.md`.

## Historical Summary

Status: **MODEL STAGE M3 IN PROGRESS - HARDWARE NUMERIC FIX REQUIRED**.

One-token alignment is not closed. The known dim1 4-ULP final mismatch is stable, H8/H9 remain identical, and the first proven divergent arithmetic operation is in the common FFN W2 reduction path.

No RTL, checkpoint, tokenizer, or hardware working-tree file was modified by this model task.
