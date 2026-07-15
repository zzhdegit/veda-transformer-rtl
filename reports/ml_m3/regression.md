# ML-M3 Regression

| Check | Result | Notes |
|---|---|---|
| Q2 checkpoint/tokenizer/export audit | PASS | `D:\IC_Workspace\VEDA_artifacts\ml_m3\manifests\artifact_audit.json` |
| Required vector generation | PASS | lengths [1, 2, 8, 16, 32] |
| Python compile for new M3 scripts | PASS | `python -m py_compile ...` completed before RTL run |
| H8/H9 one-token RTL smoke | FAIL | CHECK_FAIL layer token=0 dim=1 got=3d4a2576 expected=3d4a2572 |
| Multi-token RTL | BLOCKED | Not run after one-token gate failure |
| PDK/STA/P&R/PPA | NOT_RUN | Outside ML-M3 and not invoked |
