# Hardware Stage H9 Acceptance Audit

Status: not accepted.

## Closed Items

- Fixed workspace stayed at `D:/IC_Workspace/VEDA`.
- Branch stayed at `hw/h9-sfu-pe-interleaving`.
- H8 baseline remains the accepted hardware baseline.
- Paper schedule evidence and repository design decisions are documented.
- H9 native full-array mapping model covers D_HEAD=8, 16, 64, and 128.
- Bounded score buffer and probability FIFO RTL modules exist.
- H9 Python model tests pass.
- H9 vs H8 host bit-model comparison remains bit-exact for D_HEAD=8, 16, 64,
  and 128.
- Matched single-head RTL A/B evidence exists for paper staged versus paper
  interleaved using the same top, inputs, wrappers, and ready environment.
- Matched RTL seq16 and seq32 performance objective is met for D_HEAD=8, 16,
  and 64 in the existing A/B report.
- Performance attribution is corrected: the gain is native full-array mapping
  plus interleaving, not pure interleaving.
- `model/attention/paper_interleaved_cycle_model.py` is calibrated to the
  matched RTL counter interval for D_HEAD=8, 16, and 64 at seq 1, 2, 8, 16,
  32, and 64.
- Stage5 and Stage7D testbenches now expose `ATTENTION_SCHEDULE`, allowing
  `PAPER_ARRAY+INTERLEAVED` multi-head and full-layer runs.
- `scripts/sim/gen_stage5_vectors.py` now supports configurable
  `--max-seq-len` and config lists for H9 long-sequence/cache-full coverage.
- `scripts/sim/run_hw_h9_vcs.sh` now includes H9 multi-head, H9 full-layer,
  H9 long-sequence/cache-full, and assertion-enabled VCS entries.
- H9 Make targets include cycle calibration, multi-head, full-layer, reset,
  backpressure, cache-full, and assertion aliases.
- Host `python scripts/sim/run_hw_h9_tests.py` passes in this closure turn.
- Static H9 hygiene lint passes in this closure turn.

## Current Tool-blocked Results

The current execution environment does not provide the RTL/EDA tools required
for final acceptance:

```text
reports/hw_h9/rtl_sim_current_env.txt:
vcs: NOT FOUND
result=FAIL

reports/hw_h9/lint_results_current_env.txt:
VCS/vlogan lint compile: SKIPPED - vlogan executable not found.
result=PASS

reports/hw_h9/synth_check_current_env.txt:
dc_shell: not found
DW_FOUNDATION_SLDB: not found
DC elaboration: SKIPPED - dc_shell not found.
result=FAIL
```

## Open Items Blocking Hardware Stage H9 PASS

- Newly added H9 multi-head interleaved RTL runs have not executed because VCS
  is unavailable.
- Newly added H9 full-layer interleaved RTL runs have not executed because VCS
  is unavailable.
- Full sequence/cache-full H9 RTL coverage is wired but not executed.
- Full reset interrupt matrix is still incomplete and not executed.
- Broad deterministic/random backpressure matrix with at least 20 seeds is not
  implemented or executed.
- Assertion execution matrix is documented, but the current environment cannot
  compile/run the assertions.
- DC structural check cannot be rerun in this environment.
- Stage5/6/7/8 full RTL regressions cannot be rerun in this environment.
- No H9 accepted tag has been created.

## Decision

Do not write `HARDWARE STAGE H9 PASS`.

Do not create `hw-h9-sfu-pe-interleaving-accepted`.

Do not enter Hardware Stage H10.

Stage 8 remains the accepted hardware baseline until the remaining H9 RTL,
assertion, random-backpressure, reset, cache-full, lint/vlogan, DC, and
regression conditions are actually executed and pass.
