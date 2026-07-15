# Hardware Stage H9 Reset Results

Scope: reset interrupt matrix for HW-H9 interleaved paper Attention.

reset_injection_points=64
reset_pass_count=64
reset_failures=0
matrix=reports/hw_h9/reset_execution_matrix.md
result=PASS

Coverage note: the 64 labels execute in `tb_h9_reset_matrix` against the direct
H9 interleaved Attention datapath. Upper-layer labels are proxy labels mapped to
reachable datapath states. Independent multi-head and full-layer reset injection
wrappers are still required before Hardware Stage H9 can be accepted.
