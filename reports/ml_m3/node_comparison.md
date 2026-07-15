# ML-M3 Node Comparison

Final layer output is the first checked real-RTL data boundary in the M3 testbench. The first observable data mismatch occurs at token 0, dimension 1.

- First mismatch: `CHECK_FAIL layer token=0 dim=1 got=3d4a2576 expected=3d4a2572`
- Direct internal data-node comparison was not expanded after the one-token final-output gate failed.
- Read-only hierarchical boundary monitors were compiled into the model-line testbench, but no PASS boundary summary was emitted because the simulation stopped at the first output mismatch.
