# Hardware Stage H9B Full-Array Mapping

Status: checkpoint implemented, acceptance incomplete.

Implemented model mapping:

```text
group  = dimension % 2
local  = dimension / 2
row    = local % 8
column = local / 8
```

Model evidence:

- `tb/model/test_hw_h9_interleaved_attention.py`: PASS on host.
- D_HEAD=8 no longer maps to low cell indices 0..7.
- D_HEAD=16 uses both groups and all rows.
- D_HEAD=64 uses both groups, all rows, and multiple columns.
- D_HEAD=128 is structurally covered by model tests.

RTL smoke evidence:

- `tb_h9_single_head` D_HEAD=8: PASS, `group0=408`, `group1=408`.
- `tb_h9_single_head` D_HEAD=16: PASS, `group0=408`, `group1=408`.
- `tb_h9_single_head` D_HEAD=64: PASS, `group0=408`, `group1=408`.
- H9 DC hierarchy checks count 128 `paper_pe_cell` occurrences in checked paper
  interleaved tops.

Open acceptance coverage:

- Multi-head interleaved RTL mapping coverage.
- Full-layer interleaved RTL mapping coverage.
- Full requested sequence/tail/cache-full matrix.
