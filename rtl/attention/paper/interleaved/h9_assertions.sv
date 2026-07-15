`default_nettype none

module h9_interleaved_assertions #(
    parameter int INDEX_W = 16,
    parameter int COUNTER_W = 64
) (
    input  logic                 clk,
    input  logic                 rst_n,

    input  logic                 inner_active,
    input  logic                 outer_active,
    input  logic                 mode_switch_request,
    input  logic                 inflight_operation,
    input  logic                 outer_start,
    input  logic                 qk_retired,
    input  logic                 softmax_valid,
    input  logic                 new_head_start,
    input  logic                 previous_head_retired,

    input  logic                 score_push,
    input  logic                 score_pop,
    input  logic                 score_full,
    input  logic                 score_empty,
    input  logic                 score_valid,
    input  logic                 score_ready,
    input  logic [255:0]         score_payload,
    input  logic [INDEX_W-1:0]   score_index,
    input  logic [INDEX_W-1:0]   previous_score_index,
    input  logic [INDEX_W:0]     score_produced_count,
    input  logic [INDEX_W:0]     score_consumed_count,
    input  logic [INDEX_W:0]     score_expected_count,
    input  logic                 score_head_done,

    input  logic                 probability_push,
    input  logic                 probability_pop,
    input  logic                 probability_full,
    input  logic                 probability_empty,
    input  logic                 probability_valid,
    input  logic                 probability_ready,
    input  logic [255:0]         probability_payload,
    input  logic [INDEX_W-1:0]   probability_index,
    input  logic [INDEX_W-1:0]   previous_probability_index,
    input  logic [INDEX_W-1:0]   v_index,
    input  logic [INDEX_W:0]     probability_produced_count,
    input  logic [INDEX_W:0]     probability_consumed_count,
    input  logic [INDEX_W:0]     probability_expected_count,
    input  logic                 probability_head_done,

    input  logic                 sv_update_fire,
    input  logic                 sv_update_seen,
    input  logic                 missing_sv_update,
    input  logic                 head_done_fire,
    input  logic                 head_done_seen,
    input  logic                 cache_commit_fire,
    input  logic                 cache_commit_seen,
    input  logic [INDEX_W:0]     valid_seq_len,

    input  logic                 reset_check_valid,
    input  logic [INDEX_W:0]     reset_score_occupancy,
    input  logic [INDEX_W:0]     reset_probability_occupancy,
    input  logic                 reset_inflight,
    input  logic                 reset_ghost_valid,

    input  logic                 transaction_start,
    input  logic                 transaction_done,
    input  logic [INDEX_W:0]     transaction_start_count,
    input  logic [INDEX_W:0]     transaction_done_count,
    input  logic                 active,
    input  logic                 legal_stall,
    input  logic                 progress_event,
    input  logic                 watchdog_trigger,
    input  logic [255:0]         active_control_bus
);
`ifndef SYNTHESIS
    property p_no_inner_and_outer_same_cycle;
        @(posedge clk) disable iff (!rst_n) !(inner_active && outer_active);
    endproperty
    a_no_inner_and_outer_same_cycle: assert property (p_no_inner_and_outer_same_cycle)
        else $error("h9_assertion no_inner_and_outer_same_cycle failed");

    property p_no_mode_switch_with_inflight_operation;
        @(posedge clk) disable iff (!rst_n) !(mode_switch_request && inflight_operation);
    endproperty
    a_no_mode_switch_with_inflight_operation: assert property (p_no_mode_switch_with_inflight_operation)
        else $error("h9_assertion no_mode_switch_with_inflight_operation failed");

    property p_no_outer_before_qk_retired;
        @(posedge clk) disable iff (!rst_n) outer_start |-> qk_retired;
    endproperty
    a_no_outer_before_qk_retired: assert property (p_no_outer_before_qk_retired)
        else $error("h9_assertion no_outer_before_qk_retired failed");

    property p_no_outer_before_softmax_valid;
        @(posedge clk) disable iff (!rst_n) outer_start |-> softmax_valid;
    endproperty
    a_no_outer_before_softmax_valid: assert property (p_no_outer_before_softmax_valid)
        else $error("h9_assertion no_outer_before_softmax_valid failed");

    property p_no_new_head_before_previous_retired;
        @(posedge clk) disable iff (!rst_n) new_head_start |-> previous_head_retired;
    endproperty
    a_no_new_head_before_previous_retired: assert property (p_no_new_head_before_previous_retired)
        else $error("h9_assertion no_new_head_before_previous_retired failed");

    property p_score_count_conserved;
        @(posedge clk) disable iff (!rst_n)
            score_head_done |-> ((score_produced_count == score_expected_count) &&
                                 (score_consumed_count == score_expected_count));
    endproperty
    a_score_count_conserved: assert property (p_score_count_conserved)
        else $error("h9_assertion score_count_conserved failed");

    property p_probability_count_conserved;
        @(posedge clk) disable iff (!rst_n)
            probability_head_done |-> ((probability_produced_count == probability_expected_count) &&
                                       (probability_consumed_count == probability_expected_count));
    endproperty
    a_probability_count_conserved: assert property (p_probability_count_conserved)
        else $error("h9_assertion probability_count_conserved failed");

    property p_no_score_overflow;
        @(posedge clk) disable iff (!rst_n) !(score_push && score_full);
    endproperty
    a_no_score_overflow: assert property (p_no_score_overflow)
        else $error("h9_assertion no_score_overflow failed");

    property p_no_score_underflow;
        @(posedge clk) disable iff (!rst_n) !(score_pop && score_empty);
    endproperty
    a_no_score_underflow: assert property (p_no_score_underflow)
        else $error("h9_assertion no_score_underflow failed");

    property p_no_probability_overflow;
        @(posedge clk) disable iff (!rst_n) !(probability_push && probability_full);
    endproperty
    a_no_probability_overflow: assert property (p_no_probability_overflow)
        else $error("h9_assertion no_probability_overflow failed");

    property p_no_probability_underflow;
        @(posedge clk) disable iff (!rst_n) !(probability_pop && probability_empty);
    endproperty
    a_no_probability_underflow: assert property (p_no_probability_underflow)
        else $error("h9_assertion no_probability_underflow failed");

    property p_score_payload_stable_until_ready;
        @(posedge clk) disable iff (!rst_n)
            $past(score_valid && !score_ready) |-> (score_valid && $stable(score_payload));
    endproperty
    a_score_payload_stable_until_ready: assert property (p_score_payload_stable_until_ready)
        else $error("h9_assertion score_payload_stable_until_ready failed");

    property p_probability_payload_stable_until_ready;
        @(posedge clk) disable iff (!rst_n)
            $past(probability_valid && !probability_ready) |->
                (probability_valid && $stable(probability_payload));
    endproperty
    a_probability_payload_stable_until_ready: assert property (p_probability_payload_stable_until_ready)
        else $error("h9_assertion probability_payload_stable_until_ready failed");

    property p_score_index_monotonic;
        @(posedge clk) disable iff (!rst_n)
            score_push && (score_produced_count != '0) |-> (score_index >= previous_score_index);
    endproperty
    a_score_index_monotonic: assert property (p_score_index_monotonic)
        else $error("h9_assertion score_index_monotonic failed");

    property p_probability_index_monotonic;
        @(posedge clk) disable iff (!rst_n)
            probability_push && (probability_produced_count != '0) |->
                (probability_index >= previous_probability_index);
    endproperty
    a_probability_index_monotonic: assert property (p_probability_index_monotonic)
        else $error("h9_assertion probability_index_monotonic failed");

    property p_probability_matches_v_index;
        @(posedge clk) disable iff (!rst_n) probability_pop |-> (probability_index == v_index);
    endproperty
    a_probability_matches_v_index: assert property (p_probability_matches_v_index)
        else $error("h9_assertion probability_matches_v_index failed");

    property p_no_duplicate_sv_update;
        @(posedge clk) disable iff (!rst_n) !(sv_update_fire && sv_update_seen);
    endproperty
    a_no_duplicate_sv_update: assert property (p_no_duplicate_sv_update)
        else $error("h9_assertion no_duplicate_sv_update failed");

    property p_no_missing_sv_update;
        @(posedge clk) disable iff (!rst_n) !(probability_head_done && missing_sv_update);
    endproperty
    a_no_missing_sv_update: assert property (p_no_missing_sv_update)
        else $error("h9_assertion no_missing_sv_update failed");

    property p_no_duplicate_head_done;
        @(posedge clk) disable iff (!rst_n) !(head_done_fire && head_done_seen);
    endproperty
    a_no_duplicate_head_done: assert property (p_no_duplicate_head_done)
        else $error("h9_assertion no_duplicate_head_done failed");

    property p_no_duplicate_cache_commit;
        @(posedge clk) disable iff (!rst_n) !(cache_commit_fire && cache_commit_seen);
    endproperty
    a_no_duplicate_cache_commit: assert property (p_no_duplicate_cache_commit)
        else $error("h9_assertion no_duplicate_cache_commit failed");

    property p_valid_seq_len_changes_only_by_commit;
        @(posedge clk) disable iff (!rst_n)
            (valid_seq_len != $past(valid_seq_len)) |-> cache_commit_fire;
    endproperty
    a_valid_seq_len_changes_only_by_commit: assert property (p_valid_seq_len_changes_only_by_commit)
        else $error("h9_assertion valid_seq_len_changes_only_by_commit failed");

    property p_reset_clears_interleaved_state;
        @(posedge clk) disable iff (!rst_n)
            reset_check_valid |-> ((reset_score_occupancy == '0) &&
                                   (reset_probability_occupancy == '0) &&
                                   !reset_inflight && !reset_ghost_valid);
    endproperty
    a_reset_clears_interleaved_state: assert property (p_reset_clears_interleaved_state)
        else $error("h9_assertion reset_clears_interleaved_state failed");

    property p_no_unknown_control_when_active;
        @(posedge clk) disable iff (!rst_n) active |-> !$isunknown(active_control_bus);
    endproperty
    a_no_unknown_control_when_active: assert property (p_no_unknown_control_when_active)
        else $error("h9_assertion no_unknown_control_when_active failed");

    property p_transaction_count_conserved;
        @(posedge clk) disable iff (!rst_n)
            transaction_done |-> (transaction_done_count <= transaction_start_count);
    endproperty
    a_transaction_count_conserved: assert property (p_transaction_count_conserved)
        else $error("h9_assertion transaction_count_conserved failed");

    property p_progress_or_legal_stall;
        @(posedge clk) disable iff (!rst_n)
            watchdog_trigger |-> (progress_event || legal_stall);
    endproperty
    a_progress_or_legal_stall: assert property (p_progress_or_legal_stall)
        else $error("h9_assertion progress_or_legal_stall failed");
`endif
endmodule

`default_nettype wire
