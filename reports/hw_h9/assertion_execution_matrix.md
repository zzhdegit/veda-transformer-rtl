# Hardware Stage H9 Assertion Execution Matrix

Status: assertions compiled and executed in passing VCS runs, but the matrix is
not complete enough for final acceptance.

H9 VCS commands compile with:

```text
-assert svaext
```

The following H9 RTL simulations completed with `assertion_markers=0`:

- score buffer;
- probability FIFO;
- single-head D_HEAD 8, 16, and 64;
- matched A/B staged and interleaved D_HEAD 8, 16, and 64 at seq 1, 2, 8, 16,
  32, and 64;
- deterministic backpressure matched A/B subset;
- H9 multi-head Stage 5 wrapper runs;
- H9 full-layer Stage 7D wrapper runs.

Execution matrix:

| Assertion or check | Location | Executed top/testbench | Status |
|---|---|---|---|
| no_inner_and_outer_same_cycle | `paper_interleaved_attention_datapath.sv` | H9 single-head and matched A/B | executed, no failure |
| no_mode_switch_with_inflight_operation | `paper_interleaved_attention_datapath.sv` command/mode checks | H9 single-head and matched A/B | executed, no failure |
| no_outer_before_qk_retired | `paper_interleaved_attention_datapath.sv` outer command guard | H9 single-head and matched A/B | executed, no failure |
| no_outer_before_softmax_valid | `paper_interleaved_attention_datapath.sv` phase/softmax guard | H9 single-head and matched A/B | executed, no failure |
| no_new_head_before_previous_retired | Stage 5 multi-head scoreboards | H9 multi-head | scoreboard executed, no named SVA |
| score_count_conserved | H9 matched A/B and Stage 5 scoreboards | matched A/B and multi-head | scoreboard executed |
| probability_count_conserved | H9 matched A/B and Stage 5 scoreboards | matched A/B and multi-head | scoreboard executed |
| no_score_overflow | `paper_score_buffer.sv` | H9 score buffer and single-head | executed, no failure |
| no_score_underflow | `paper_score_buffer.sv` | H9 score buffer and single-head | executed, no failure |
| no_probability_overflow | `paper_probability_fifo.sv` | H9 probability FIFO and single-head | executed, no failure |
| no_probability_underflow | `paper_probability_fifo.sv` | H9 probability FIFO and single-head | executed, no failure |
| score_payload_stable_until_ready | `paper_score_buffer.sv` | H9 score buffer | executed, no failure |
| probability_payload_stable_until_ready | `paper_probability_fifo.sv` | H9 probability FIFO | executed, no failure |
| probability_matches_v_index | `paper_interleaved_attention_datapath.sv` | H9 single-head and matched A/B | executed, no failure |
| no_duplicate_sv_update | H9 output/count checks | H9 single-head and matched A/B | scoreboard executed |
| no_missing_sv_update | H9 output/count checks | H9 single-head and matched A/B | scoreboard executed |
| no_duplicate_head_done | Stage 5 multi-head done checks | H9 multi-head | scoreboard executed |
| no_duplicate_cache_commit | Stage 5 multi-head commit checks | H9 multi-head/cache-full | scoreboard executed |
| valid_seq_len_changes_only_by_commit | Stage 5 expected seq checks | H9 multi-head/cache-full | scoreboard executed |
| reset_clears_interleaved_state | Stage 5 reset checks plus RTL reset logic | H9 multi-head | partially executed |
| progress_or_legal_stall | watchdogs in H9/Stage5/Stage7 testbenches | H9 single/multi/layer | executed for deterministic runs |

Why this is not final acceptance evidence:

- several requested names are not explicit named SV assertion properties;
- no negative tests or bind-based assertion activation proof were added;
- the full reset matrix and 20-seed random backpressure tests are absent.

Therefore assertions are exercised, but Hardware Stage H9 remains not accepted.
