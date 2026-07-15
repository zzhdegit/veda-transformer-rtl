`timescale 1ns/1ps
`default_nettype none

module tb_h9_assertion_negative;
    localparam int INDEX_W = 8;
    localparam int COUNTER_W = 64;

    logic clk;
    logic rst_n;
    string negative_case;

    logic inner_active;
    logic outer_active;
    logic mode_switch_request;
    logic inflight_operation;
    logic outer_start;
    logic qk_retired;
    logic softmax_valid;
    logic new_head_start;
    logic previous_head_retired;
    logic score_push;
    logic score_pop;
    logic score_full;
    logic score_empty;
    logic score_valid;
    logic score_ready;
    logic [255:0] score_payload;
    logic [INDEX_W-1:0] score_index;
    logic [INDEX_W-1:0] previous_score_index;
    logic [INDEX_W:0] score_produced_count;
    logic [INDEX_W:0] score_consumed_count;
    logic [INDEX_W:0] score_expected_count;
    logic score_head_done;
    logic probability_push;
    logic probability_pop;
    logic probability_full;
    logic probability_empty;
    logic probability_valid;
    logic probability_ready;
    logic [255:0] probability_payload;
    logic [INDEX_W-1:0] probability_index;
    logic [INDEX_W-1:0] previous_probability_index;
    logic [INDEX_W-1:0] v_index;
    logic [INDEX_W:0] probability_produced_count;
    logic [INDEX_W:0] probability_consumed_count;
    logic [INDEX_W:0] probability_expected_count;
    logic probability_head_done;
    logic sv_update_fire;
    logic sv_update_seen;
    logic missing_sv_update;
    logic head_done_fire;
    logic head_done_seen;
    logic cache_commit_fire;
    logic cache_commit_seen;
    logic [INDEX_W:0] valid_seq_len;
    logic reset_check_valid;
    logic [INDEX_W:0] reset_score_occupancy;
    logic [INDEX_W:0] reset_probability_occupancy;
    logic reset_inflight;
    logic reset_ghost_valid;
    logic transaction_start;
    logic transaction_done;
    logic [INDEX_W:0] transaction_start_count;
    logic [INDEX_W:0] transaction_done_count;
    logic active;
    logic legal_stall;
    logic progress_event;
    logic watchdog_trigger;
    logic [255:0] active_control_bus;

    h9_interleaved_assertions #(
        .INDEX_W(INDEX_W),
        .COUNTER_W(COUNTER_W)
    ) u_assertions (
        .clk(clk),
        .rst_n(rst_n),
        .inner_active(inner_active),
        .outer_active(outer_active),
        .mode_switch_request(mode_switch_request),
        .inflight_operation(inflight_operation),
        .outer_start(outer_start),
        .qk_retired(qk_retired),
        .softmax_valid(softmax_valid),
        .new_head_start(new_head_start),
        .previous_head_retired(previous_head_retired),
        .score_push(score_push),
        .score_pop(score_pop),
        .score_full(score_full),
        .score_empty(score_empty),
        .score_valid(score_valid),
        .score_ready(score_ready),
        .score_payload(score_payload),
        .score_index(score_index),
        .previous_score_index(previous_score_index),
        .score_produced_count(score_produced_count),
        .score_consumed_count(score_consumed_count),
        .score_expected_count(score_expected_count),
        .score_head_done(score_head_done),
        .probability_push(probability_push),
        .probability_pop(probability_pop),
        .probability_full(probability_full),
        .probability_empty(probability_empty),
        .probability_valid(probability_valid),
        .probability_ready(probability_ready),
        .probability_payload(probability_payload),
        .probability_index(probability_index),
        .previous_probability_index(previous_probability_index),
        .v_index(v_index),
        .probability_produced_count(probability_produced_count),
        .probability_consumed_count(probability_consumed_count),
        .probability_expected_count(probability_expected_count),
        .probability_head_done(probability_head_done),
        .sv_update_fire(sv_update_fire),
        .sv_update_seen(sv_update_seen),
        .missing_sv_update(missing_sv_update),
        .head_done_fire(head_done_fire),
        .head_done_seen(head_done_seen),
        .cache_commit_fire(cache_commit_fire),
        .cache_commit_seen(cache_commit_seen),
        .valid_seq_len(valid_seq_len),
        .reset_check_valid(reset_check_valid),
        .reset_score_occupancy(reset_score_occupancy),
        .reset_probability_occupancy(reset_probability_occupancy),
        .reset_inflight(reset_inflight),
        .reset_ghost_valid(reset_ghost_valid),
        .transaction_start(transaction_start),
        .transaction_done(transaction_done),
        .transaction_start_count(transaction_start_count),
        .transaction_done_count(transaction_done_count),
        .active(active),
        .legal_stall(legal_stall),
        .progress_event(progress_event),
        .watchdog_trigger(watchdog_trigger),
        .active_control_bus(active_control_bus)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    task automatic clear_signals;
        begin
            inner_active = 1'b0;
            outer_active = 1'b0;
            mode_switch_request = 1'b0;
            inflight_operation = 1'b0;
            outer_start = 1'b0;
            qk_retired = 1'b1;
            softmax_valid = 1'b1;
            new_head_start = 1'b0;
            previous_head_retired = 1'b1;
            score_push = 1'b0;
            score_pop = 1'b0;
            score_full = 1'b0;
            score_empty = 1'b0;
            score_valid = 1'b0;
            score_ready = 1'b1;
            score_payload = 256'h1;
            score_index = '0;
            previous_score_index = '0;
            score_produced_count = '0;
            score_consumed_count = '0;
            score_expected_count = '0;
            score_head_done = 1'b0;
            probability_push = 1'b0;
            probability_pop = 1'b0;
            probability_full = 1'b0;
            probability_empty = 1'b0;
            probability_valid = 1'b0;
            probability_ready = 1'b1;
            probability_payload = 256'h2;
            probability_index = '0;
            previous_probability_index = '0;
            v_index = '0;
            probability_produced_count = '0;
            probability_consumed_count = '0;
            probability_expected_count = '0;
            probability_head_done = 1'b0;
            sv_update_fire = 1'b0;
            sv_update_seen = 1'b0;
            missing_sv_update = 1'b0;
            head_done_fire = 1'b0;
            head_done_seen = 1'b0;
            cache_commit_fire = 1'b0;
            cache_commit_seen = 1'b0;
            valid_seq_len = '0;
            reset_check_valid = 1'b0;
            reset_score_occupancy = '0;
            reset_probability_occupancy = '0;
            reset_inflight = 1'b0;
            reset_ghost_valid = 1'b0;
            transaction_start = 1'b0;
            transaction_done = 1'b0;
            transaction_start_count = '0;
            transaction_done_count = '0;
            active = 1'b0;
            legal_stall = 1'b0;
            progress_event = 1'b1;
            watchdog_trigger = 1'b0;
            active_control_bus = '0;
        end
    endtask

    task automatic pulse_violation(input string name);
        begin
            if (name == "no_inner_and_outer_same_cycle") begin
                inner_active = 1'b1;
                outer_active = 1'b1;
            end else if (name == "no_mode_switch_with_inflight_operation") begin
                mode_switch_request = 1'b1;
                inflight_operation = 1'b1;
            end else if (name == "no_outer_before_qk_retired") begin
                outer_start = 1'b1;
                qk_retired = 1'b0;
            end else if (name == "no_outer_before_softmax_valid") begin
                outer_start = 1'b1;
                softmax_valid = 1'b0;
            end else if (name == "no_new_head_before_previous_retired") begin
                new_head_start = 1'b1;
                previous_head_retired = 1'b0;
            end else if (name == "score_count_conserved") begin
                score_head_done = 1'b1;
                score_expected_count = 9'd4;
                score_produced_count = 9'd4;
                score_consumed_count = 9'd3;
            end else if (name == "probability_count_conserved") begin
                probability_head_done = 1'b1;
                probability_expected_count = 9'd4;
                probability_produced_count = 9'd4;
                probability_consumed_count = 9'd3;
            end else if (name == "no_score_overflow") begin
                score_push = 1'b1;
                score_full = 1'b1;
            end else if (name == "no_score_underflow") begin
                score_pop = 1'b1;
                score_empty = 1'b1;
            end else if (name == "no_probability_overflow") begin
                probability_push = 1'b1;
                probability_full = 1'b1;
            end else if (name == "no_probability_underflow") begin
                probability_pop = 1'b1;
                probability_empty = 1'b1;
            end else if (name == "score_payload_stable_until_ready") begin
                score_valid = 1'b1;
                score_ready = 1'b0;
                @(posedge clk);
                score_payload = 256'hbad0;
            end else if (name == "probability_payload_stable_until_ready") begin
                probability_valid = 1'b1;
                probability_ready = 1'b0;
                @(posedge clk);
                probability_payload = 256'hbad1;
            end else if (name == "probability_matches_v_index") begin
                probability_pop = 1'b1;
                probability_index = 8'd3;
                v_index = 8'd2;
            end else if (name == "no_duplicate_sv_update") begin
                sv_update_fire = 1'b1;
                sv_update_seen = 1'b1;
            end else if (name == "no_missing_sv_update") begin
                probability_head_done = 1'b1;
                missing_sv_update = 1'b1;
            end else if (name == "no_duplicate_head_done") begin
                head_done_fire = 1'b1;
                head_done_seen = 1'b1;
            end else if (name == "no_duplicate_cache_commit") begin
                cache_commit_fire = 1'b1;
                cache_commit_seen = 1'b1;
            end else if (name == "valid_seq_len_changes_only_by_commit") begin
                valid_seq_len = 9'd1;
                @(posedge clk);
                valid_seq_len = 9'd2;
                cache_commit_fire = 1'b0;
            end else if (name == "reset_clears_interleaved_state") begin
                reset_check_valid = 1'b1;
                reset_score_occupancy = 9'd1;
                reset_probability_occupancy = 9'd1;
                reset_ghost_valid = 1'b1;
            end else if (name == "no_unknown_control_when_active") begin
                active = 1'b1;
                active_control_bus = {255'd0, 1'bx};
            end else if (name == "transaction_count_conserved") begin
                transaction_done = 1'b1;
                transaction_start_count = 9'd1;
                transaction_done_count = 9'd2;
            end else if (name == "progress_or_legal_stall") begin
                watchdog_trigger = 1'b1;
                progress_event = 1'b0;
                legal_stall = 1'b0;
            end else begin
                $display("HW_H9_ASSERTION_NEGATIVE_FAIL unknown case=%s", name);
                $fatal(1);
            end
        end
    endtask

    initial begin
        if (!$value$plusargs("NEGATIVE_CASE=%s", negative_case)) begin
            negative_case = "no_inner_and_outer_same_cycle";
        end
        rst_n = 1'b0;
        clear_signals();
        repeat (3) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);
        pulse_violation(negative_case);
        @(posedge clk);
        clear_signals();
        repeat (4) @(posedge clk);
        $display("HW_H9_ASSERTION_NEGATIVE_DONE case=%s", negative_case);
        $finish;
    end
endmodule

`default_nettype wire
