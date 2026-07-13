# Hardware Stage H9 Summary

Status: closure checkpoint complete, not accepted.

Completed so far:

- Fixed workspace and branch setup.
- H8 baseline confirmed as PASS.
- Paper schedule evidence frozen.
- H9 spec and stream protocol added.
- H9 native full-array mapping model added.
- H9 interleaved softmax/reference/cycle models added.
- H9 model tests pass on host and Docker.
- H9 bounded score buffer and probability FIFO RTL tests pass.
- H9 interleaved paper single-head RTL smoke tests pass for D_HEAD=8, 16, and
  64.
- H9 single-head smoke tests report actual nonzero QK-SFU and SFU-sV overlap
  counters.
- H9 matched RTL A/B baseline now passes for single-head paper staged versus
  paper interleaved using the same `single_head_attention` top, inputs,
  DesignWare latency, and ready environment.
- Matched RTL seq16/seq32 performance objective is met for D_HEAD=8, 16, and
  64. Example no-backpressure totals:
  - D8 seq16/32: 1363/2707 staged versus 1169/2209 interleaved.
  - D16 seq16/32: 2472/4920 staged versus 1171/2211 interleaved.
  - D64 seq16/32: 9126/18198 staged versus 1183/2223 interleaved.
- The previous structural Python cycle comparison is documented as not
  apples-to-apples for performance acceptance.
- H9 lint/vlogan passes with only accepted DesignWare pragma-no-effect warnings.
- H9 DC structural checks pass for legacy staged, paper staged, and paper
  interleaved selections with 128 `paper_pe_cell` occurrences in checked paper
  interleaved tops.
- Stage 5/6/7/8 regressions pass after the H9 checkpoint changes.

Still pending for Hardware Stage H9 acceptance:

- H9 multi-head interleaved RTL testbench coverage.
- H9 full-layer interleaved RTL testbench coverage.
- Exhaustive H9 reset, random-backpressure, cache-full extra-token, and
  long-sequence RTL coverage.
- Cycle model equations are not yet calibrated to reproduce the matched RTL
  counter intervals exactly.
- Final acceptance tag.

Do not write `HARDWARE STAGE H9 PASS` until the open items in
`reports/hw_h9/acceptance_audit.md` are closed.
