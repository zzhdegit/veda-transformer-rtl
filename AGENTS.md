# Agent Rules

1. Before starting work, read `AGENTS.md`, `PROJECT_STATE.md`, `HANDOFF.md`, the current stage Markdown file, and the specification files directly required by the current stage.
2. Repository files are the only trusted source. Conversation history is not a specification.
3. Authoritative specifications are `PROJECT_STATE.md`, `HANDOFF.md`, the active stage file under `transformer_rtl_plan_md/`, and the explicit spec files under `docs/`. Tool notes or legacy backend notes are not stage specifications unless one of those authoritative files names them.
4. Do not silently change data formats, bit widths, rounding, module interfaces, pipeline latency, ready/valid protocol, K/V storage layout, or Transformer structure.
5. If one of those items must change, first document the reason in `PROJECT_STATE.md`, mark it as pending confirmation, and do not assume approval.
6. Every functional change must update the corresponding tests.
7. Do not use `real` or `shortreal` in formal synthesizable datapaths, hide failing tests, warnings, X values, or timing violations, claim valid area from large flip-flop arrays standing in for SRAM, or invent synthesis, power, timing, or layout results.
8. PDKs, standard-cell libraries, and EDA installation directories must stay outside the repository.
9. At the end of every stage, update `PROJECT_STATE.md`, `HANDOFF.md`, tests, and report summaries.
10. `HANDOFF.md` must state what was completed, what was not completed, dependencies, reproduction steps, and next-stage cautions.
11. Do not run destructive Git operations or overwrite user-owned files.

## GitHub Remote Workflow

- Public repository: https://github.com/zzhdegit/veda-transformer-rtl
- Remote name: origin
- Default branch: main
- Machine-local details are recorded in ignored LOCAL_GITHUB.md.
- Authentication uses Windows Git Credential Manager.
- Run git status and git fetch origin before remote modifications.
- Use fast-forward-only updates when possible.
- Never request, print, export, persist, or commit authentication tokens.
- Never place credentials in remote URLs.
- Never force push without explicit user approval.
- Never commit Synopsys DesignWare source, PDK, licenses, technology libraries,
  SRAM macros, generated simulation files, waveforms, or EDA installation files.
