# ML-M3 RTL Smoke Results

Status: **MODEL STAGE M3 IN PROGRESS - HARDWARE NUMERIC FIX REQUIRED**

- H8/D8/D_MODEL64/D_FFN256 compile/elaborate: PASS.
- H8 staged one-token diagnostic: FAIL versus bit model, H8/H9 capture identical.
- H9 interleaved one-token diagnostic: FAIL versus bit model, H8/H9 capture identical.
- First checked final-output difference: `CHECK_FAIL layer token=0 dim=1 got=3d4a2576 expected=3d4a2572`.
- Full 64-dimension diagnostic mismatches: 54.
- H8/H9 diagnostic capture SHA256: `d09c4b80953c18ba7edea335490cca63f5ac4d9fcf24f4e17027813ca0045fb4`.
- Original fail-fast captured prefix SHA256 for both schedules: `5adbf7b5ef5e5fbff1a767e271d852ab711ec9d829a9f7fe9125288901d4f3be`.

Diagnostic logs:

```text
D:/IC_Workspace/VEDA_artifacts/ml_m3/rtl_logs/ml_m3_staged_len_1_numeric_alignment_edge_adds.log
D:/IC_Workspace/VEDA_artifacts/ml_m3/rtl_logs/ml_m3_interleaved_len_1_numeric_alignment_edge_adds.log
```

Because one-token smoke still fails bit-exact comparison, ML-M3 did not enter
length 2/8/16 RTL co-simulation.
