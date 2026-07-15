# ML-M3 Hardware Dependency Issue

The accepted H9 hardware baseline compiles for the Q2 H8/D8 layer configuration, but real-weight one-token final output is not bit-exact against the current hardware-aware bit model.

- First mismatch: `CHECK_FAIL layer token=0 dim=1 got=3d4a2576 expected=3d4a2572`
- H8 staged log: `D:\IC_Workspace\VEDA_artifacts\ml_m3\rtl_logs\ml_m3_staged_len_1_smoke_len1_combined.log`
- H9 interleaved log: `D:\IC_Workspace\VEDA_artifacts\ml_m3\rtl_logs\ml_m3_interleaved_len_1_smoke_len1_combined.log`
- Vector: `D:\IC_Workspace\VEDA_artifacts\ml_m3\vectors\len_1\case_len_1.mem`
- Hardware repo was not modified by this task.
