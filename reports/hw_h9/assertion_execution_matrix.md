# Hardware Stage H9 Assertion Execution Matrix

Status: assertions exist and H9 scripts compile with `-assert svaext` when VCS
is available; execution is blocked in the current environment.

| Assertion or check | Location | Intended top/testbench | Current execution |
|---|---|---|---|
| no_inner_and_outer_same_cycle | `paper_interleaved_attention_datapath.sv` | H9 single-head and matched A/B | not executed: `vcs` not found |
| no_mode_switch_with_inflight_operation | `paper_interleaved_attention_datapath.sv` command/mode checks | H9 single-head and matched A/B | not executed |
| no_outer_before_qk_retired | `paper_interleaved_attention_datapath.sv` outer command guard | H9 single-head and matched A/B | not executed |
| no_outer_before_softmax_valid | `paper_interleaved_attention_datapath.sv` phase/softmax guard | H9 single-head and matched A/B | not executed |
| no_new_head_before_previous_retired | `multi_head_generation_engine.sv` output/cache commit checks | H9 multi-head | not executed |
| score_count_conserved | H9 testbench output/count checks | matched A/B and multi-head | not executed |
| probability_count_conserved | H9 testbench output/count checks | matched A/B and multi-head | not executed |
| no_score_overflow | `paper_score_buffer.sv` | H9 buffer and single-head | not executed |
| no_score_underflow | `paper_score_buffer.sv` | H9 buffer and single-head | not executed |
| no_probability_overflow | `paper_probability_fifo.sv` | H9 buffer and single-head | not executed |
| no_probability_underflow | `paper_probability_fifo.sv` | H9 buffer and single-head | not executed |
| score_payload_stable_until_ready | `paper_score_buffer.sv` | H9 buffer and single-head | not executed |
| probability_payload_stable_until_ready | `paper_probability_fifo.sv` | H9 buffer and single-head | not executed |
| probability_matches_v_index | `paper_interleaved_attention_datapath.sv` | H9 single-head and matched A/B | not executed |
| no_duplicate_sv_update | H9 testbench output/count checks | H9 single-head | not executed |
| no_missing_sv_update | H9 testbench output/count checks | H9 single-head | not executed |
| no_duplicate_head_done | `multi_head_generation_engine.sv` done/cache checks | H9 multi-head | not executed |
| no_duplicate_cache_commit | `multi_head_generation_engine.sv` cache commit checks | H9 multi-head | not executed |
| valid_seq_len_changes_only_by_commit | Stage 5 testbench expected seq checks | H9 multi-head/cache-full | not executed |
| reset_clears_interleaved_state | Stage 5 reset tests plus RTL reset logic | H9 multi-head | not executed |
| progress_or_legal_stall | watchdogs in H9/Stage5/Stage7 testbenches | H9 single/multi/layer | not executed |

Current compile/run status:

```text
reports/hw_h9/rtl_sim_current_env.txt:
vcs: NOT FOUND
result=FAIL
```

This matrix is not acceptance evidence. It is the execution plan plus current
tool-blocked status.
