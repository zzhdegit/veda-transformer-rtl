# Stage 2 Summary

Status: STAGE 2 PASS.

Stage 2 implements and verifies a correctness-first reconfigurable PE compute
core for GEMV/inner-product and outer-product tile transactions. The design uses
the frozen Stage 1B `fp16_to_fp32` and `fp32_mac_wrapper` interfaces and adds a
new `fp32_add_wrapper` boundary around local DesignWare `DW_fp_add`.

## Input Baseline

- Branch: `stage2-pe-core`.
- Base commit at Stage 2 start: `db0f062`.
- Stage 1B artifacts were present in the working tree but not committed; they
  matched `HANDOFF.md` and passed Stage 1B regression before and after Stage 2.
- Docker container: `nailong`.
- VCS: `O-2018.09-SP2-2_Full64`.
- DC: `L-2016.03-SP1`.
- No PDK, standard-cell target library, SRAM macro, P&R, formal STA, area,
  power, WNS, or frequency result was used or generated.

## Implemented Modules

- `rtl/arithmetic/fp32_add_wrapper.sv`
- `rtl/pe/lane_mask_generator.sv`
- `rtl/pe/pe_lane.sv`
- `rtl/pe/fp32_reduction_tree.sv`
- `rtl/pe/accumulator_bank.sv`
- `rtl/pe/pe_perf_counter.sv`
- `rtl/pe/reconfigurable_pe_core.sv`

Python bit models:

- `model/arithmetic/fp32_add_reference.py`
- `model/pe/pe_lane_reference.py`
- `model/pe/reduction_tree_reference.py`
- `model/pe/pe_core_reference.py`

## Architecture

`reconfigurable_pe_core` accepts one tile transaction at a time:

- mode: `MODE_GEMV`, `MODE_QK_INNER`, or `MODE_SV_OUTER`
- clear, `tile_first`, `tile_last`
- explicit lane mask or active-lane count generated mask
- FP32 scalar for outer-product
- packed FP16 lane vectors
- metadata and `last`

The internal path is:

```text
input transaction register
-> lane mask
-> FP16-to-FP32 conversion
-> PE lanes through fp32_mac_wrapper
-> time-multiplexed balanced FP32 reduction tree through fp32_add_wrapper
-> inner tile accumulator or outer accumulator bank
-> output transaction register
-> performance counters
```

DesignWare native FP instances appear only inside:

- `fp32_mac_wrapper`
- `fp32_add_wrapper`

## Supported Modes

- `MODE_QK_INNER`: computes `score_i = sum_j q_j * K_i,j`.
- `MODE_GEMV`: aliases the same inner-product datapath for one output element.
- `MODE_SV_OUTER`: updates `acc_j = p_i * V_i,j + acc_j` for active lanes.

Full Softmax, Score Buffer, KV Cache Manager, Attention Controller, multi-head
attention, Transformer layer, Voting, P&R, and PPA are not implemented.

## Reduction And Tiling Order

The reduction tree order is fixed and bit-model matched:

```text
level 0: p0, p1, p2, ..., pN-1
level 1: p0+p1, p2+p3, ...
level 2: sum0+sum1, sum2+sum3, ...
...
```

`PE_NUM` must be a power of two; the RTL asserts this at elaboration. Active
lane counts do not need to be powers of two. Masked lanes are converted to FP32
zero before product/reduction.

For `D_HEAD > PE_NUM`, tiles are processed in arrival order:

```text
tile_sum_t = balanced_sum(products_t)
score = (((0 + tile_sum_0) + tile_sum_1) + ...)
```

The last tile mask is supplied by the input transaction. `tile_first` clears the
inner accumulator; `tile_last` produces the final scalar output.

## Outer Feedback Rule

Outer-product uses the local accumulator bank. Active lanes update only after the
PE lane FMA result is accepted; inactive lanes hold their previous accumulator
value. The core does not accept the next transaction until the current update is
committed or final output is drained. This is a correct feedback baseline, not
an II=1 outer engine.

## Latency And II

- `fp16_to_fp32`: latency 1, II 1, inherited from Stage 1B.
- `fp32_mac_wrapper`: latency 1, II 1, inherited from Stage 1B.
- `fp32_add_wrapper`: latency 1, II 1.
- `pe_lane`: latency 1, II 1 when used as an isolated lane.
- `fp32_reduction_tree`: one vector at a time; fixed balanced order is
  time-multiplexed through one `fp32_add_wrapper`; service time is proportional
  to `2 * (PE_NUM - 1)` control cycles plus input/output handoff.
- `reconfigurable_pe_core`: one tile transaction at a time; top-level tile II is
  the transaction service time. Outer-product top-level II is not 1 in this
  correctness baseline.

## Verification Results

Host:

- `python scripts/sim/run_stage2_tests.py`: PASS.
- `python -m pytest tb/model/test_stage1b_fp.py tb/model/test_stage2_pe.py`:
  12 passed.
- Python `py_compile`: PASS.

Docker:

- `make stage1b-rtl-sim`: PASS.
- `make stage1b-lint`: PASS.
- `make stage1b-synth`: PASS.
- `make stage2-rtl-sim`: PASS.
  - `fp32_add_wrapper`: PASS, 30 vectors.
  - `fp32_add_invalid`: PASS expected assertion failure.
  - `pe_lane`: PASS.
  - `fp32_reduction_tree`: PASS.
  - `reconfigurable_pe_core`: PASS.
- `make stage2-lint`: PASS; vlogan diagnostics none.
- `make stage2-synth`: PASS; DC analyze/elaborate/link/check_design passed for
  default PE_NUM=8 and parameterized PE_NUM=128.

## Performance Counter Result

The directed Stage 2 core simulation reported:

```text
total_cycles=100
busy_cycles=90
active_lane_cycles=42
available_lane_cycles=56
input_stall_cycles=67
output_stall_cycles=1
mode_switch_cycles=2
tile_count=7
operation_count=4
invalid_count=0
```

Utilization is defined as:

```text
valid PE lane FP operations / available PE lane operation slots
```

For the directed core run this is `42 / 56 = 75%`. This is a functional RTL
cycle statistic only, not a frequency, area, power, timing, or PPA claim.

## Known Limitations

- The reduction tree is a time-multiplexed balanced topology, not a parallel or
  deeply pipelined physical tree.
- The PE core is transaction-serial for correctness and backpressure closure.
- The current Stage 1B/2 FP wrappers use combinational DesignWare arithmetic
  paths before output registers; these paths are likely future critical paths.
- Real physical pipelining, retiming, and PPA require a target library and
  timing constraints.
- Stage 1B remains uncommitted in the local working tree even though it is the
  verified baseline for this Stage 2 branch.

## Conclusion

STAGE 2 PASS.

Correctness architecture is accepted. Physical pipeline is provisional. Stage 3
may start from this tile transaction interface, but it must not assume PE core
top-level II=1 or any PPA result.
