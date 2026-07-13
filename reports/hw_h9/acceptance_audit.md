# Hardware Stage H9 Acceptance Audit

Status: not accepted.

Closed checkpoint items:

- Fixed workspace stayed at `D:/IC_Workspace/VEDA`.
- Branch `hw/h9-sfu-pe-interleaving` was created and pushed.
- H8 baseline was confirmed PASS before H9 work.
- Paper schedule evidence and repository design decisions were documented.
- H9 native full-array mapping model covers D_HEAD=8, 16, 64, and 128
  structure.
- H9 Python model tests pass.
- H9 RTL score buffer and probability FIFO tests pass.
- H9 interleaved paper single-head RTL smoke tests pass for D_HEAD=8, 16, and
  64.
- H9 RTL smoke counters show nonzero QK-SFU overlap and SFU-sV overlap.
- H9 lint/vlogan passes.
- H9 DC analyze/elaborate/link/check_design passes; no PPA is claimed.
- Stage 5/6/7/8 regressions pass after the H9 checkpoint changes.

Open items blocking Hardware Stage H9 PASS:

- Multi-head interleaved RTL testbench is not implemented.
- Full-layer interleaved RTL testbench is not implemented.
- Required H9 sequence set 1/2/3/7/8/9/15/16/31/32/MAX_SEQ_LEN and
  cache-full extra-token coverage is incomplete.
- Exhaustive H9 reset interrupt matrix is incomplete.
- Deterministic and random H9 backpressure matrix is incomplete.
- Random seeds have not been saved for broad H9 backpressure/deadlock testing.
- H9 RTL is not yet proven against the H9 bit model across all required
  configurations.
- The current H9 cycle model is faster than the full-array non-interleaved H9
  schedule, but slower than the H8 staged paper baseline for seq 1/2/8/16/32.
- No H9 accepted tag has been created.

Do not write `HARDWARE STAGE H9 PASS` until all HW-H9 exit conditions are closed.
