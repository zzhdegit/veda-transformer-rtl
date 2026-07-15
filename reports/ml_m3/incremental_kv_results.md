# ML-M3 Incremental KV Results

Status: **BLOCKED BY ONE-TOKEN NUMERIC GATE**.

The mandatory one-token RTL smoke still fails bit-exact comparison against the
hardware-aware bit model. Per the ML-M3 Numeric Alignment task boundary,
length 2/8/16 RTL incremental KV co-simulation was not run after identifying
the common RTL W2 reduction-path numeric blocker.

Software full-vs-incremental reference from the earlier artifact audit remains
PASS, but real RTL incremental KV closure must wait for the hardware numeric
fix and a passing one-token H8/H9 smoke.
