# Hardware Stage H9 Multi-Head Random Backpressure Results

Scope: broad fixed-seed random stress on the real multi_head_generation_engine hierarchy.
Watchdog formula in testbench: N_HEAD * token_count * (D_HEAD + token_count) * 250 + 50000.
Endpoint mask: token_valid_gap, multi_head_output_ready, multi_head_done_ready, real_head_boundary_pressure, real_commit_near_pressure.

HW_H9_MULTI_HEAD_RANDOM_PASS seed=101 config=H2/D8 seq=8 token_count=2 pattern=1 cycles=1211 watchdog=60000 source_gap=16 output_stall=3 done_stall=2 simultaneous=589 head_boundary_stall=2 commit_near_stall=2 score_peak=2 prob_peak=2 commits=2 outputs=4 done=2
HW_H9_MULTI_HEAD_RANDOM_PASS seed=211 config=H2/D8 seq=8 token_count=2 pattern=2 cycles=1206 watchdog=60000 source_gap=5 output_stall=0 done_stall=0 simultaneous=0 head_boundary_stall=1 commit_near_stall=0 score_peak=2 prob_peak=2 commits=2 outputs=4 done=2
HW_H9_MULTI_HEAD_RANDOM_PASS seed=307 config=H2/D8 seq=8 token_count=2 pattern=3 cycles=1220 watchdog=60000 source_gap=98 output_stall=6 done_stall=8 simultaneous=469 head_boundary_stall=2 commit_near_stall=2 score_peak=2 prob_peak=2 commits=2 outputs=4 done=2
HW_H9_MULTI_HEAD_RANDOM_PASS seed=401 config=H2/D8 seq=8 token_count=2 pattern=4 cycles=1214 watchdog=60000 source_gap=88 output_stall=3 done_stall=5 simultaneous=548 head_boundary_stall=2 commit_near_stall=2 score_peak=2 prob_peak=2 commits=2 outputs=4 done=2
HW_H9_MULTI_HEAD_RANDOM_PASS seed=503 config=H2/D8 seq=8 token_count=8 pattern=5 cycles=9500 watchdog=114000 source_gap=161 output_stall=13 done_stall=7 simultaneous=3232 head_boundary_stall=7 commit_near_stall=6 score_peak=2 prob_peak=2 commits=8 outputs=16 done=8
HW_H9_MULTI_HEAD_RANDOM_PASS seed=601 config=H2/D8 seq=8 token_count=8 pattern=2 cycles=9484 watchdog=114000 source_gap=20 output_stall=2 done_stall=2 simultaneous=3 head_boundary_stall=7 commit_near_stall=6 score_peak=2 prob_peak=2 commits=8 outputs=16 done=8
HW_H9_MULTI_HEAD_RANDOM_PASS seed=701 config=H2/D8 seq=8 token_count=8 pattern=3 cycles=9504 watchdog=114000 source_gap=450 output_stall=16 done_stall=8 simultaneous=3667 head_boundary_stall=8 commit_near_stall=7 score_peak=2 prob_peak=2 commits=8 outputs=16 done=8
HW_H9_MULTI_HEAD_RANDOM_PASS seed=809 config=H2/D8 seq=8 token_count=8 pattern=4 cycles=9549 watchdog=114000 source_gap=201 output_stall=48 done_stall=21 simultaneous=4426 head_boundary_stall=6 commit_near_stall=6 score_peak=2 prob_peak=2 commits=8 outputs=16 done=8
HW_H9_MULTI_HEAD_RANDOM_PASS seed=907 config=H2/D8 seq=16 token_count=16 pattern=5 cycles=31445 watchdog=242000 source_gap=349 output_stall=31 done_stall=38 simultaneous=10758 head_boundary_stall=15 commit_near_stall=12 score_peak=2 prob_peak=3 commits=16 outputs=32 done=16
HW_H9_MULTI_HEAD_RANDOM_PASS seed=1009 config=H2/D8 seq=16 token_count=16 pattern=2 cycles=31380 watchdog=242000 source_gap=41 output_stall=2 done_stall=2 simultaneous=5 head_boundary_stall=15 commit_near_stall=14 score_peak=2 prob_peak=3 commits=16 outputs=32 done=16
HW_H9_MULTI_HEAD_RANDOM_PASS seed=1103 config=H2/D8 seq=16 token_count=16 pattern=3 cycles=31445 watchdog=242000 source_gap=921 output_stall=44 done_stall=25 simultaneous=12202 head_boundary_stall=14 commit_near_stall=14 score_peak=2 prob_peak=3 commits=16 outputs=32 done=16
HW_H9_MULTI_HEAD_RANDOM_PASS seed=1201 config=H2/D8 seq=16 token_count=16 pattern=4 cycles=31494 watchdog=242000 source_gap=425 output_stall=72 done_stall=46 simultaneous=14656 head_boundary_stall=12 commit_near_stall=13 score_peak=2 prob_peak=3 commits=16 outputs=32 done=16
HW_H9_MULTI_HEAD_RANDOM_PASS seed=1301 config=H4/D8 seq=8 token_count=8 pattern=5 cycles=19011 watchdog=178000 source_gap=378 output_stall=35 done_stall=16 simultaneous=6464 head_boundary_stall=17 commit_near_stall=8 score_peak=2 prob_peak=2 commits=8 outputs=32 done=8
HW_H9_MULTI_HEAD_RANDOM_PASS seed=1409 config=H4/D8 seq=8 token_count=8 pattern=2 cycles=18969 watchdog=178000 source_gap=40 output_stall=8 done_stall=1 simultaneous=3 head_boundary_stall=20 commit_near_stall=8 score_peak=2 prob_peak=2 commits=8 outputs=32 done=8
HW_H9_MULTI_HEAD_RANDOM_PASS seed=1511 config=H4/D8 seq=8 token_count=8 pattern=3 cycles=19071 watchdog=178000 source_gap=686 output_stall=103 done_stall=8 simultaneous=7375 head_boundary_stall=21 commit_near_stall=7 score_peak=2 prob_peak=2 commits=8 outputs=32 done=8
HW_H9_MULTI_HEAD_RANDOM_PASS seed=1601 config=H4/D8 seq=8 token_count=8 pattern=4 cycles=19060 watchdog=178000 source_gap=380 output_stall=85 done_stall=15 simultaneous=8845 head_boundary_stall=17 commit_near_stall=7 score_peak=2 prob_peak=2 commits=8 outputs=32 done=8
HW_H9_MULTI_HEAD_RANDOM_PASS seed=1709 config=H2/D16 seq=8 token_count=8 pattern=5 cycles=12262 watchdog=146000 source_gap=433 output_stall=51 done_stall=11 simultaneous=4109 head_boundary_stall=6 commit_near_stall=8 score_peak=2 prob_peak=2 commits=8 outputs=32 done=8
HW_H9_MULTI_HEAD_RANDOM_PASS seed=1801 config=H2/D16 seq=8 token_count=8 pattern=2 cycles=12207 watchdog=146000 source_gap=38 output_stall=6 done_stall=1 simultaneous=2 head_boundary_stall=8 commit_near_stall=8 score_peak=2 prob_peak=2 commits=8 outputs=32 done=8
HW_H9_MULTI_HEAD_RANDOM_PASS seed=1907 config=H2/D16 seq=8 token_count=8 pattern=3 cycles=12265 watchdog=146000 source_gap=766 output_stall=65 done_stall=0 simultaneous=4712 head_boundary_stall=7 commit_near_stall=7 score_peak=2 prob_peak=2 commits=8 outputs=32 done=8
HW_H9_MULTI_HEAD_RANDOM_PASS seed=2003 config=H2/D16 seq=8 token_count=8 pattern=4 cycles=12277 watchdog=146000 source_gap=287 output_stall=66 done_stall=11 simultaneous=5668 head_boundary_stall=7 commit_near_stall=6 score_peak=2 prob_peak=2 commits=8 outputs=32 done=8
HW_H9_MULTI_HEAD_RANDOM_PASS seed=2111 config=H1/D64 seq=8 token_count=8 pattern=5 cycles=14362 watchdog=194000 source_gap=725 output_stall=91 done_stall=11 simultaneous=4708 head_boundary_stall=0 commit_near_stall=8 score_peak=2 prob_peak=2 commits=8 outputs=64 done=8
HW_H9_MULTI_HEAD_RANDOM_PASS seed=2203 config=H1/D64 seq=8 token_count=8 pattern=2 cycles=14281 watchdog=194000 source_gap=73 output_stall=19 done_stall=2 simultaneous=2 head_boundary_stall=0 commit_near_stall=8 score_peak=2 prob_peak=2 commits=8 outputs=64 done=8
HW_H9_MULTI_HEAD_RANDOM_PASS seed=2309 config=H1/D64 seq=8 token_count=8 pattern=3 cycles=14356 watchdog=194000 source_gap=1562 output_stall=96 done_stall=0 simultaneous=5412 head_boundary_stall=0 commit_near_stall=7 score_peak=2 prob_peak=2 commits=8 outputs=64 done=8
HW_H9_MULTI_HEAD_RANDOM_PASS seed=2411 config=H1/D64 seq=8 token_count=8 pattern=4 cycles=14394 watchdog=194000 source_gap=919 output_stall=114 done_stall=20 simultaneous=6495 head_boundary_stall=0 commit_near_stall=8 score_peak=2 prob_peak=2 commits=8 outputs=64 done=8

run_count=24
pass_count=24
failures=0
matrix=reports/hw_h9/multi_head_random_backpressure_matrix.md
result=PASS
