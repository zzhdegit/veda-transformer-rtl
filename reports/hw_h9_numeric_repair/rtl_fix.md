# HW-H9-N1 RTL Fix

## RTL Changes

- `rtl/arithmetic/fp32_add_wrapper.sv`
  - Changed DesignWare add rounding mode from `3'b100` to `3'b000`.
  - Added an elaboration-time guard that fails if the RNE constant is not
    `3'b000`.

- `rtl/pe/fp32_reduction_tree.sv`
  - Added simulation-only inflight tracking for add launch pair, width, and
    operands.
  - Added assertions for result/pair/width association and stalled payload
    stability.

No production debug port was added. No external RTL interface changed.

## Testbench and Script Changes

- `tb/rtl/hw_h9_numeric/tb_hw_h9_numeric_repair.sv`
  - New numeric repair testbench for wrapper, reduction tree, and PE core.

- `scripts/sim/gen_hw_h9_numeric_repair_vectors.py`
  - Generates known operand, directed pair/tile, boundary, and random vectors.

- `scripts/sim/run_hw_h9_numeric_repair_vcs.sh`
- `scripts/sim/run_hw_h9_numeric_repair.py`
- `scripts/sim/run_hw_h9_q2_length1_vcs.sh`
- `scripts/sim/run_hw_h9_q2_length1.py`
  - New committed regression drivers.

- `tb/rtl/stage7/tb_stage7d_transformer_layer.sv`
  - Added large-vector configurability and output-stall mode control for the
    external real-Q2 one-token run.

- `Makefile`
  - Added `hw-h9-numeric-repair` and `hw-h9-q2-length1` targets.
