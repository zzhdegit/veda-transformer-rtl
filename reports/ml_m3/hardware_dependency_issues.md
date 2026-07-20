# ML-M3 Hardware Dependency Issue

## Closure Note

This is a preserved historical blocker report from the previous H9 thesis hardware baseline. The blocker is closed on `hw-h9-real-weight-numeric-repair-accepted` at commit `a54e608a8dc7e63c7e5dd342f8b893bb1e0b7485`.

Current acceptance status: `MODEL STAGE M3 PASS`. See `reports/ml_m3/acceptance_audit.md` and `reports/ml_m3/hardware_readonly_audit.md` for the repaired-baseline audit.

## Historical Blocker

The accepted H9 hardware baseline compiles for the Q2 H8/D8 layer configuration, but real-weight one-token final output is not bit-exact against the current hardware-aware bit model.

- First mismatch: `CHECK_FAIL layer token=0 dim=1 got=3d4a2576 expected=3d4a2572`
- H8 staged log: `D:\IC_Workspace\VEDA_artifacts\ml_m3\rtl_logs\ml_m3_staged_len_1_smoke_len1_combined.log`
- H9 interleaved log: `D:\IC_Workspace\VEDA_artifacts\ml_m3\rtl_logs\ml_m3_interleaved_len_1_smoke_len1_combined.log`
- Vector: `D:\IC_Workspace\VEDA_artifacts\ml_m3\vectors\len_1\case_len_1.mem`
- Hardware repo was not modified by this task.
