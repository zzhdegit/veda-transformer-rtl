`default_nettype none

bind paper_interleaved_attention_datapath h9_interleaved_assertions #(
    .INDEX_W(SEQ_LEN_W),
    .COUNTER_W(COUNTER_W)
) u_h9_interleaved_assertions (
    .clk                            (clk),
    .rst_n                          (rst_n),
    .inner_active                   (qk_active),
    .outer_active                   (sv_active),
    .mode_switch_request            (phase_q == PH_NORM_START),
    .inflight_operation             (array_cmd_valid || array_result_valid || array_done_valid ||
                                     qk_active || sv_active),
    .outer_start                    (array_cmd_fire && (array_cmd_mode == MODE_OUTER_PRODUCT)),
    .qk_retired                     ((phase_q == PH_OUTER) || (phase_q == PH_OUTPUT) ||
                                     (phase_q == PH_DONE)),
    .softmax_valid                  (softmax_final_seen_q),
    .new_head_start                 (start_fire),
    .previous_head_retired          (phase_q == PH_IDLE),

    .score_push                     (scaler_out_fire),
    .score_pop                      (reduction_in_fire),
    .score_full                     (score_fifo_occupancy == seq_len_q),
    .score_empty                    (score_fifo_occupancy == '0),
    .score_valid                    (scaler_out_valid),
    .score_ready                    (scaler_out_ready),
    .score_payload                  ({{(256-SEQ_LEN_W-32){1'b0}}, qk_token_q, scaler_out_score}),
    .score_index                    (qk_token_q),
    .previous_score_index           (qk_token_q),
    .score_produced_count           ({1'b0, score_valid_count_q}),
    .score_consumed_count           ({1'b0, score_valid_count_q}),
    .score_expected_count           ({1'b0, score_valid_count_q}),
    .score_head_done                ((phase_q == PH_OUTER) || (phase_q == PH_OUTPUT) ||
                                     (phase_q == PH_DONE)),

    .probability_push               (norm_prob_fire),
    .probability_pop                (array_cmd_fire && (array_cmd_mode == MODE_OUTER_PRODUCT)),
    .probability_full               (prob_fifo_occupancy == seq_len_q),
    .probability_empty              (prob_fifo_occupancy == '0),
    .probability_valid              (norm_prob_valid),
    .probability_ready              (norm_prob_ready),
    .probability_payload            ({{(256-TOKEN_W-32){1'b0}}, norm_prob_index, norm_prob_value}),
    .probability_index              (SEQ_LEN_W'(norm_prob_index)),
    .previous_probability_index     (SEQ_LEN_W'(norm_prob_index)),
    .v_index                        (sv_token_q),
    .probability_produced_count     ({1'b0, prob_valid_count_q}),
    .probability_consumed_count     (((phase_q == PH_OUTPUT) || (phase_q == PH_DONE) ||
                                      ((phase_q == PH_OUTER) && (sv_state_q == SV_DONE))) ?
                                      {1'b0, seq_len_q} : {1'b0, sv_token_q}),
    .probability_expected_count     ({1'b0, seq_len_q}),
    .probability_head_done          ((phase_q == PH_OUTPUT) || (phase_q == PH_DONE)),

    .sv_update_fire                 (array_cmd_fire && (array_cmd_mode == MODE_OUTER_PRODUCT)),
    .sv_update_seen                 (1'b0),
    .missing_sv_update              (1'b0),
    .head_done_fire                 (done_fire),
    .head_done_seen                 (1'b0),
    .cache_commit_fire              (1'b0),
    .cache_commit_seen              (1'b0),
    .valid_seq_len                  ('0),

    .reset_check_valid              ((phase_q == PH_IDLE) && !active_q),
    .reset_score_occupancy          ({1'b0, score_fifo_occupancy}),
    .reset_probability_occupancy    ({1'b0, prob_fifo_occupancy}),
    .reset_inflight                 (array_cmd_valid || array_result_valid || array_done_valid ||
                                     scaler_in_valid || scaler_out_valid || reduction_in_valid ||
                                     reduction_final_valid || norm_score_valid || norm_prob_valid),
    .reset_ghost_valid              (output_valid || done_valid),

    .transaction_start              (start_fire),
    .transaction_done               (done_fire),
    .transaction_start_count        ({{SEQ_LEN_W{1'b0}}, 1'b1}),
    .transaction_done_count         ({{SEQ_LEN_W{1'b0}}, done_fire}),
    .active                         (active_q),
    .legal_stall                    ((output_valid && !output_ready) || (done_valid && !done_ready) ||
                                     (array_cmd_valid && !array_cmd_ready)),
    .progress_event                 (array_cmd_fire || array_result_fire || array_done_fire ||
                                     scaler_in_fire || scaler_out_fire || reduction_in_fire ||
                                     reduction_final_fire || norm_score_fire || norm_prob_fire ||
                                     output_fire || done_fire),
    .watchdog_trigger               (1'b0),
    .active_control_bus             ({248'd0,
                                      phase_q[0], qk_state_q[0], sv_state_q[0],
                                      qk_token_q[0], reduce_token_q[0], norm_token_q[0],
                                      sv_token_q[0], invalid_q})
);

`default_nettype wire
