# Hardware Stage H9 Schedule Optimization

Status: no datapath schedule optimization applied in this closure checkpoint.

The first closure priority was a fair matched RTL A/B. That audit changed the
performance conclusion without changing the H9 datapath arithmetic or
controller schedule:

- Added `tb_h9_matched_ab_single_head.sv`.
- Added staged and interleaved matched runs to `run_hw_h9_vcs.sh`.
- Added the missing staged-path `sram_2p_wrapper.sv` filelist dependency.
- Kept Softmax arithmetic, FP16/FP32 boundaries, reduction order,
  current-token semantics, and cache semantics unchanged.

Because matched RTL already satisfies the seq16/seq32 performance gate for the
single-head D_HEAD=8/16/64 configs, no risky schedule rewrite was made in this
checkpoint.

Remaining optimization candidates, if later required by multi-head/full-layer
closure:

- expose H9 overlap counters as formal top-level performance ports instead of
  testbench-only hierarchical reads;
- remove residual fixed stream setup bubbles while preserving packet stability;
- add true internal ready controls for score/SFU/probability random
  backpressure tests;
- calibrate the Python cycle model against the matched RTL counter intervals.
