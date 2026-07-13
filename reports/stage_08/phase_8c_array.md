# Stage 8C Independent Paper Array Report

## Result

Stage 8C independent paper-structured 8x8x2 PE array: PASS

Date: 2026-07-13

Branch: `stage8-paper-pe-array`

This phase implements and verifies the independent PE array only. Attention
QK/sV integration is not part of this phase and is not claimed complete here.

## Added RTL

- `rtl/pe/paper/paper_pe_cell.sv`
- `rtl/pe/paper/paper_l1_reduction.sv`
- `rtl/pe/paper/paper_l2_reduction.sv`
- `rtl/pe/paper/paper_pe_group.sv`
- `rtl/pe/paper/paper_array_8x8x2.sv`

The RTL preserves the repository mixed-precision arithmetic contract:

- FP16 operands are expanded through `fp16_to_fp32`.
- MAC operations use `fp32_mac_wrapper`.
- Reduction operations use `fp32_add_wrapper`.
- Partial sums and accumulators are FP32.
- No DesignWare primitive is instantiated directly outside existing wrappers.

## Array Structure

| Item | Stage 8C RTL |
|---|---:|
| PE groups | 2 |
| Rows per group | 8 |
| Columns per group | 8 |
| Physical PE cells | 128 |
| Type-A PE cells | 64 |
| Type-B PE cells | 64 |

`paper_pe_group.sv` uses explicit nested generate loops for 8 rows and
8 columns. `paper_array_8x8x2.sv` instantiates two groups. Type-A uses even
columns and Type-B uses odd columns, matching the Stage 8A repository decision.

## Interface

The independent array uses the command plus stream-style interface frozen in
Stage 8A:

- command: `cmd_valid`, `cmd_ready`, `cmd_mode`, sizes, tile id, metadata,
  clear/tile/group/lane masks, scalar, A/B operands, and last.
- result: `result_valid`, `result_ready`, mode, tile id, scalar/vector result,
  lane mask, status, invalid, metadata, and last.
- done: `done_valid`, `done_ready`, status, invalid, and metadata.
- counters: paper-array active/idle cycles, inner/outer mode cycles,
  group active cycles, tail masked PE cycles, mode switch cycles, and
  input/output stall cycles.

## RTL Verification

Testbench: `tb/rtl/stage8/tb_paper_array_inner.sv`

Covered cases:

- inner-product lengths 1, 7, 8, 9, 15, 16, 31, 32, 64, and 128;
- one group active and both groups active;
- cancellation to +0;
- outer-product mode with vector output;
- mode switch from inner to outer;
- output backpressure and done backpressure;
- reset during an active inner-product transaction followed by a clean token;
- result metadata, tile id, invalid/status, stable result payload, and done.

Run result:

```text
STAGE8C_PAPER_ARRAY_PASS active=55 inner=55 outer=0 group0=36 group1=36 tail=127 mode_switch=1 output_stall=3
paper_array result=PASS run_exit_code=0 assertion_markers=0
```

## Commands

Host:

- `python scripts/lint/run_stage8c_lint.py`: PASS for static checks; vlogan
  skipped because host `vlogan` is not available.

Docker:

- `docker exec nailong bash -lc 'cd /workspace/VEDA && make stage8c-test'`: PASS
- `docker exec nailong bash -lc 'cd /workspace/VEDA && make stage8c-lint'`: PASS
- `docker exec nailong bash -lc 'cd /workspace/VEDA && make stage8c-rtl-sim'`: PASS
- `docker exec nailong bash -lc 'cd /workspace/VEDA && make stage8c-synth'`: PASS

Generated reports:

- `reports/stage_08/phase_8c_lint_results.txt`
- `reports/stage_08/phase_8c_vcs_rtl_sim.txt`
- `reports/stage_08/phase_8c_synth_check.txt`
- `reports/stage_08/phase_8c_dc_elaborate.log`
- `reports/stage_08/dc_hierarchy_stage8c_paper_array_8x8x2.rpt`
- `reports/stage_08/dc_check_stage8c_paper_array_8x8x2.rpt`

## Lint And DC

Static hygiene: PASS.

VCS/vlogan compile lint in Docker:

```text
vlogan_exit_code=0
vlogan_diagnostics: none
```

DC analyze/elaborate/link/check_design:

```text
DC elaboration result: PASS
```

DC check_design emits structural warnings after elaboration, dominated by
constant zero connections, unloaded underflow/status outputs, unconnected
metadata outputs inside wrappers, and unmapped GTECH reporting. These are
recorded in `dc_check_stage8c_paper_array_8x8x2.rpt`. They are not used as
area, timing, power, frequency, or physical implementation conclusions.

## Limitations

- Stage 8C is an independent PE-array checkpoint; Attention QK/sV is not yet
  mapped to this array.
- The outer-product test covers one deterministic vector case, not the full
  Attention sV adapter.
- DC is used only for structural elaboration and hierarchy counting.
- Cycle counters are RTL counters from this focused testbench only; they are
  not a throughput or PPA conclusion.
- SFU-PE interleaving remains out of scope.
- KV cache eviction remains out of scope.
