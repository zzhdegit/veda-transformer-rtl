# Hardware Stage H9 Random Backpressure Matrix

| Item | Configuration | Testbench | Seed/Injection | Expected | Result | Log |
|---|---|---|---|---|---|---|
| 1 | H1/D8, seq1 | tb_h9_random_backpressure | seed=101 pattern=0 | no deadlock, stable payload, bit-exact output | PASS | build/hw_h9_random_backpressure/random_seed_101_d8_s1.log |
| 2 | H1/D8, seq8 | tb_h9_random_backpressure | seed=211 pattern=1 | no deadlock, stable payload, bit-exact output | PASS | build/hw_h9_random_backpressure/random_seed_211_d8_s8.log |
| 3 | H1/D8, seq16 | tb_h9_random_backpressure | seed=307 pattern=2 | no deadlock, stable payload, bit-exact output | PASS | build/hw_h9_random_backpressure/random_seed_307_d8_s16.log |
| 4 | H1/D8, seq32 | tb_h9_random_backpressure | seed=401 pattern=3 | no deadlock, stable payload, bit-exact output | PASS | build/hw_h9_random_backpressure/random_seed_401_d8_s32.log |
| 5 | H1/D16, seq16 | tb_h9_random_backpressure | seed=503 pattern=4 | no deadlock, stable payload, bit-exact output | PASS | build/hw_h9_random_backpressure/random_seed_503_d16_s16.log |
| 6 | H1/D16, seq32 | tb_h9_random_backpressure | seed=601 pattern=5 | no deadlock, stable payload, bit-exact output | PASS | build/hw_h9_random_backpressure/random_seed_601_d16_s32.log |
| 7 | H1/D64, seq16 | tb_h9_random_backpressure | seed=701 pattern=3 | no deadlock, stable payload, bit-exact output | PASS | build/hw_h9_random_backpressure/random_seed_701_d64_s16.log |
| 8 | H1/D64, seq32 | tb_h9_random_backpressure | seed=809 pattern=4 | no deadlock, stable payload, bit-exact output | PASS | build/hw_h9_random_backpressure/random_seed_809_d64_s32.log |
| 9 | H1/D8, seq8 | tb_h9_random_backpressure | seed=907 pattern=5 | no deadlock, stable payload, bit-exact output | PASS | build/hw_h9_random_backpressure/random_seed_907_d8_s8.log |
| 10 | H1/D16, seq8 | tb_h9_random_backpressure | seed=1009 pattern=2 | no deadlock, stable payload, bit-exact output | PASS | build/hw_h9_random_backpressure/random_seed_1009_d16_s8.log |
| 11 | H1/D64, seq8 | tb_h9_random_backpressure | seed=1103 pattern=1 | no deadlock, stable payload, bit-exact output | PASS | build/hw_h9_random_backpressure/random_seed_1103_d64_s8.log |
| 12 | H1/D8, seq16 | tb_h9_random_backpressure | seed=1201 pattern=4 | no deadlock, stable payload, bit-exact output | PASS | build/hw_h9_random_backpressure/random_seed_1201_d8_s16.log |
| 13 | H1/D16, seq16 | tb_h9_random_backpressure | seed=1301 pattern=5 | no deadlock, stable payload, bit-exact output | PASS | build/hw_h9_random_backpressure/random_seed_1301_d16_s16.log |
| 14 | H1/D64, seq16 | tb_h9_random_backpressure | seed=1409 pattern=3 | no deadlock, stable payload, bit-exact output | PASS | build/hw_h9_random_backpressure/random_seed_1409_d64_s16.log |
| 15 | H1/D8, seq32 | tb_h9_random_backpressure | seed=1511 pattern=2 | no deadlock, stable payload, bit-exact output | PASS | build/hw_h9_random_backpressure/random_seed_1511_d8_s32.log |
| 16 | H1/D16, seq32 | tb_h9_random_backpressure | seed=1601 pattern=1 | no deadlock, stable payload, bit-exact output | PASS | build/hw_h9_random_backpressure/random_seed_1601_d16_s32.log |
| 17 | H1/D64, seq32 | tb_h9_random_backpressure | seed=1709 pattern=5 | no deadlock, stable payload, bit-exact output | PASS | build/hw_h9_random_backpressure/random_seed_1709_d64_s32.log |
| 18 | H1/D8, seq1 | tb_h9_random_backpressure | seed=1801 pattern=0 | no deadlock, stable payload, bit-exact output | PASS | build/hw_h9_random_backpressure/random_seed_1801_d8_s1.log |
| 19 | H1/D16, seq2 | tb_h9_random_backpressure | seed=1907 pattern=3 | no deadlock, stable payload, bit-exact output | PASS | build/hw_h9_random_backpressure/random_seed_1907_d16_s2.log |
| 20 | H1/D64, seq8 | tb_h9_random_backpressure | seed=2003 pattern=4 | no deadlock, stable payload, bit-exact output | PASS | build/hw_h9_random_backpressure/random_seed_2003_d64_s8.log |
