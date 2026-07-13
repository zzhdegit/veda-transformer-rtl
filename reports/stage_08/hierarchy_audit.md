# Stage 8C Hierarchy Audit

## Result

Independent `paper_array_8x8x2` hierarchy audit: PASS

Date: 2026-07-13

Branch: `stage8-paper-pe-array`

Source:

- RTL top: `rtl/pe/paper/paper_array_8x8x2.sv`
- DC hierarchy: `reports/stage_08/dc_hierarchy_stage8c_paper_array_8x8x2.rpt`

## Instance Counts

| Instance class | Count | Evidence |
|---|---:|---|
| `paper_pe_group` | 2 | Two explicit group instances under `paper_array_8x8x2`. |
| `paper_pe_cell` | 128 | DC hierarchy contains 128 unique `paper_pe_cell_*` entries. |
| Type-A PE cells | 64 | Cell instance parameter field `..._0_16_1`. |
| Type-B PE cells | 64 | Cell instance parameter field `..._1_16_1`. |

Verification commands:

```text
rg -o "paper_pe_cell_[0-9]_[0-9]_[0-9]_[0-9]_16_1" reports/stage_08/dc_hierarchy_stage8c_paper_array_8x8x2.rpt
rg -o "paper_pe_cell_[0-9]_[0-9]_[0-9]_0_16_1" reports/stage_08/dc_hierarchy_stage8c_paper_array_8x8x2.rpt
rg -o "paper_pe_cell_[0-9]_[0-9]_[0-9]_1_16_1" reports/stage_08/dc_hierarchy_stage8c_paper_array_8x8x2.rpt
```

Observed:

```text
paper_pe_cell total = 128
Type-A total = 64
Type-B total = 64
```

## Topology Notes

- `paper_array_8x8x2.sv` instantiates two `paper_pe_group` instances with
  `group_g = 0` and `group_g = 1`.
- `paper_pe_group.sv` contains explicit nested generate loops:
  `row_g = 0..7` and `col_g = 0..7`.
- `paper_pe_cell.sv` records `GROUP_INDEX`, `ROW_INDEX`, `COLUMN_INDEX`, and
  `PE_TYPE` as parameters.
- Type-A is assigned to even columns and Type-B to odd columns.
- Inner-product mode reduces per-row products through L1 reductions and then
  group scalars through L2 reductions.
- Outer-product mode uses local PE accumulators and emits the group vectors
  without replacing the frozen Stage 5/6/7 arithmetic format.

## DC Scope

The DC run performs analyze, elaborate, link, check_design, and hierarchy
reporting only. It does not produce or imply area, power, timing, frequency,
or physical implementation results.

Known DC structural warning classes in this checkpoint:

| Warning class | Count in check_design summary | Notes |
|---|---:|---|
| LINT-28 | 50 | Unconnected metadata/status-style ports after focused structural elaboration. |
| LINT-1 | 138 | Unused generated cells after constant propagation and focused top elaboration. |
| LINT-32 | 4307 | Constants connected to wide zero/tail inputs. |
| LINT-33 | 130 | Same constant net connected to multiple pins on the same cell. |
| LINT-2 | 4774 | Unloaded nets, mostly wrapper status/underflow outputs and focused-test unused signals. |

These warnings are tracked as structural-elaboration noise for Stage 8C and do
not replace VCS/vlogan lint, which reports no diagnostics for the Stage 8C RTL
file set.
