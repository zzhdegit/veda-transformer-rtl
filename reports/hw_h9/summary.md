# Hardware Stage H9 Summary

Status: final-closure progress checkpoint, not accepted.

## Completed In This Closure Turn

- Reconfirmed work continued on `D:/IC_Workspace/VEDA` and branch
  `hw/h9-sfu-pe-interleaving`.
- Added `ATTENTION_SCHEDULE` parameter control to the Stage5 multi-head and
  Stage7D full-layer RTL testbenches without changing their default staged
  behavior.
- Added configurable `MAX_SEQ_LEN` and config-list generation to
  `scripts/sim/gen_stage5_vectors.py`.
- Extended `scripts/sim/run_hw_h9_vcs.sh` so `hw-h9-rtl-sim` now includes:
  - matched single-head staged/interleaved A/B;
  - deterministic matched backpressure subset;
  - H9 multi-head interleaved runs for H1/D8, H2/D8, H4/D8, H2/D16, and H1/D64;
  - H9 H1/D8 `MAX_SEQ_LEN=32` sequence/cache-full run;
  - H9 full-layer interleaved runs for H1/D8, H2/D8, H2/D8 two-token, H4/D8,
    and H2/D16.
- Calibrated `model/attention/paper_interleaved_cycle_model.py` to exact
  matched RTL A/B cycle formulas for D_HEAD 8, 16, and 64 at seq 1, 2, 8, 16,
  32, and 64.
- Added model tests for the calibrated RTL cycle points.
- Added H9 Make aliases for cycle calibration, multi-head, full-layer, reset,
  backpressure, cache-full, and assertion tests.
- Added final-closure reports for performance attribution, cycle calibration,
  multi-head, full-layer, sequence, cache-full, reset, random backpressure,
  assertion execution, and numerical results.

## Passing Commands In This Environment

```text
python scripts/sim/run_hw_h9_tests.py
python scripts/lint/run_hw_h9_lint.py
```

The host/model command passed 7 H9 pytest cases, H9 vs H8 bit-exact comparison,
Stage8 Python regression, Stage7A Python regression, and py_compile.

Static hygiene lint passed. VCS/vlogan compile was skipped because `vlogan` is
not installed in the current environment.

## Blocked Commands In This Environment

```text
bash scripts/sim/run_hw_h9_vcs.sh
python scripts/synth/run_hw_h9_synth_check.py
```

`run_hw_h9_vcs.sh` exits before compilation because `vcs` is not found.
`run_hw_h9_synth_check.py` fails because `dc_shell` and
`DW_FOUNDATION_SLDB` are not found.

The current-environment tool-block logs are saved separately as:

```text
reports/hw_h9/rtl_sim_current_env.txt
reports/hw_h9/lint_results_current_env.txt
reports/hw_h9/synth_check_current_env.txt
```

The checkpoint `rtl_sim.txt`, `lint_results.txt`, and `synth_check.txt` files
remain preserved as prior VCS/vlogan/DC evidence.

## Performance Status

Matched RTL A/B remains the performance authority. The current accepted
evidence is the existing `reports/hw_h9/matched_rtl_ab_baseline.md`, not the
old structural cycle model.

The calibrated model reproduces the matched RTL cycles exactly for D_HEAD 8,
16, and 64 at seq 1, 2, 8, 16, 32, and 64.

The measured speedup is the combined result of H9 native full-array mapping
plus SFU/PE interleaving. It must not be described as pure interleaving gain.

## Acceptance Status

Hardware Stage H9 remains:

```text
HW-H9 IN PROGRESS, NOT ACCEPTED
```

No accepted tag was created. Stage 8 remains the accepted hardware baseline.
