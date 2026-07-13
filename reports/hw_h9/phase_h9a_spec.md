# Hardware Stage H9A Spec Checkpoint

Status: checkpoint complete.

Completed in H9A:

- Read `AGENTS.md`, `PROJECT_STATE.md`, `HANDOFF.md`, `README.md`, Stage 8 docs/reports, paper-array RTL/model files, current attention/SFU/cache RTL, and the local VEDA paper.
- Confirmed H8 accepted baseline: `stage8-paper-array-correctness-accepted`.
- Created HW-H9 branch `hw/h9-sfu-pe-interleaving` from Stage 8 PASS.
- Froze paper evidence in `docs/hw_h9/paper_schedule_evidence.md`.
- Froze HW-H9 spec in `docs/hw_h9/spec.md`.
- Froze packet protocol in `docs/hw_h9/stream_protocol.md`.

Repository design decisions:

- H9 native mapping interleaves dimensions across groups first, then rows, then columns.
- Current Stage 3 online softmax reduction and replay normalization arithmetic is preserved.
- FIFO depths and ready/valid timing are repository-defined.
- H9 does not claim SRAM, PPA, timing closure, eviction, global PE sharing, or paper-exact RTL.

Hardware Stage H9 final acceptance remains open; this H9A checkpoint only
freezes evidence, schedule/spec, and repository design decisions.
