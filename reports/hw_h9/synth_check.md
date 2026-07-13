# Hardware Stage H9 Synth Check

Status: PASS.

Command:

```text
make hw-h9-synth
```

Scope is analyze, elaborate, link, check_design, and hierarchy checks only.

Result:

```text
dc_exit_code=0
DC elaboration result: PASS
dc_hierarchy_hw_h9_single_head_d8_interleaved.rpt paper_pe_cell_occurrences=128
dc_hierarchy_hw_h9_single_head_d64_interleaved.rpt paper_pe_cell_occurrences=128
dc_hierarchy_hw_h9_transformer_h2_d8_interleaved.rpt paper_pe_cell_occurrences=128
Checked legacy staged, paper staged, and paper interleaved architecture/schedule selections.
result=PASS
```

DC emitted parameterized base-design alias UID-109 text after successful
elaboration. The script treats those messages as tolerated only when
`dc_exit_code=0` and the expected reports are generated.

No area, power, timing, frequency, WNS, STA, layout, or PPA result is claimed.
