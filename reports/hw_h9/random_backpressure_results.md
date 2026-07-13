# Hardware Stage H9 Backpressure Results

Status: deterministic output/done backpressure exists; broad random
backpressure remains open.

Closed evidence from the matched RTL A/B checkpoint:

- no external backpressure matched A/B;
- deterministic output/done ready pattern for D_HEAD 8, 16, and 64 at seq16
  and seq32;
- bounded score/probability buffer stable-payload assertions in RTL source.

Implemented but not executed in this closure turn:

- multi-head output/done deterministic backpressure via Stage 5 testbench;
- full-layer final output backpressure via Stage 7D testbench.

Still open for acceptance:

- at least 20 fixed random seeds;
- random score/SFU/probability/array/head/final/done ready patterns;
- saved failed seed and transaction trace on failure;
- watchdog derived from maximum legal latency.

Current execution result:

```text
vcs: NOT FOUND
result=FAIL
```

No random seed list is accepted for HW-H9 Final Closure in this environment.
