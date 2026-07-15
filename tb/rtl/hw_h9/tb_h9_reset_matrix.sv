`timescale 1ns/1ps
`default_nettype none

`ifndef HW_H9_D_HEAD
`define HW_H9_D_HEAD 8
`endif

`ifndef HW_H9_SEQ_LEN
`define HW_H9_SEQ_LEN 8
`endif

module tb_h9_reset_matrix;
    localparam int PE_NUM = 8;
    localparam int D_HEAD = `HW_H9_D_HEAD;
    localparam int MAX_SEQ_LEN = 64;
    localparam int META_W = 16;
    localparam int COUNTER_W = 64;
    localparam int TOKEN_W = (MAX_SEQ_LEN <= 1) ? 1 : $clog2(MAX_SEQ_LEN);
    localparam int SEQ_LEN_W = (MAX_SEQ_LEN <= 1) ? 1 : $clog2(MAX_SEQ_LEN + 1);
    localparam int D_ADDR_W = (D_HEAD <= 1) ? 1 : $clog2(D_HEAD);
    localparam int EXPECTED_OUTPUTS = (D_HEAD + PE_NUM - 1) / PE_NUM;

    logic clk;
    logic rst_n;
    logic load_valid;
    logic load_ready;
    logic [1:0] load_kind;
    logic [TOKEN_W-1:0] load_token;
    logic [D_ADDR_W-1:0] load_dim;
    logic [15:0] load_data;
    logic start_valid;
    logic start_ready;
    logic [SEQ_LEN_W-1:0] start_seq_len;
    logic [META_W-1:0] start_meta;
    logic output_valid;
    logic output_ready;
    logic [D_ADDR_W-1:0] output_base_dim;
    logic [PE_NUM*32-1:0] output_vector_fp32;
    logic [PE_NUM-1:0] output_lane_mask;
    logic [7:0] output_status;
    logic output_invalid;
    logic [META_W-1:0] output_meta;
    logic output_last;
    logic done_valid;
    logic done_ready;
    logic [7:0] done_status;
    logic done_invalid;
    logic [META_W-1:0] done_meta;

    logic [COUNTER_W-1:0] perf_total_attention_cycles;
    logic [COUNTER_W-1:0] perf_qk_cycles;
    logic [COUNTER_W-1:0] perf_qk_pe_busy_cycles;
    logic [COUNTER_W-1:0] perf_scale_cycles;
    logic [COUNTER_W-1:0] perf_reduction_cycles;
    logic [COUNTER_W-1:0] perf_reduction_finalize_cycles;
    logic [COUNTER_W-1:0] perf_normalization_cycles;
    logic [COUNTER_W-1:0] perf_sv_cycles;
    logic [COUNTER_W-1:0] perf_pe_stall_cycles;
    logic [COUNTER_W-1:0] perf_sfu_stall_cycles;
    logic [COUNTER_W-1:0] perf_buffer_stall_cycles;
    logic [COUNTER_W-1:0] perf_output_stall_cycles;
    logic [COUNTER_W-1:0] perf_score_buffer_peak_occupancy;
    logic [COUNTER_W-1:0] perf_paper_array_active_cycles;
    logic [COUNTER_W-1:0] perf_paper_array_idle_cycles;
    logic [COUNTER_W-1:0] perf_inner_mode_cycles;
    logic [COUNTER_W-1:0] perf_outer_mode_cycles;
    logic [COUNTER_W-1:0] perf_group0_active_cycles;
    logic [COUNTER_W-1:0] perf_group1_active_cycles;
    logic [COUNTER_W-1:0] perf_tail_masked_pe_cycles;
    logic [COUNTER_W-1:0] perf_mode_switch_cycles;
    logic [COUNTER_W-1:0] perf_array_input_stall_cycles;
    logic [COUNTER_W-1:0] perf_array_output_stall_cycles;
    logic [COUNTER_W-1:0] perf_qk_sfu_overlap_cycles;
    logic [COUNTER_W-1:0] perf_qk_only_cycles;
    logic [COUNTER_W-1:0] perf_sfu_during_qk_cycles;
    logic [COUNTER_W-1:0] perf_score_fifo_full_stall_cycles;
    logic [COUNTER_W-1:0] perf_score_fifo_empty_cycles;
    logic [COUNTER_W-1:0] perf_score_fifo_peak_occupancy;
    logic [COUNTER_W-1:0] perf_sfu_sv_overlap_cycles;
    logic [COUNTER_W-1:0] perf_sfu_only_cycles;
    logic [COUNTER_W-1:0] perf_sv_only_cycles;
    logic [COUNTER_W-1:0] perf_probability_fifo_full_stall_cycles;
    logic [COUNTER_W-1:0] perf_probability_fifo_empty_stall_cycles;
    logic [COUNTER_W-1:0] perf_probability_fifo_peak_occupancy;
    logic [COUNTER_W-1:0] perf_inner_to_outer_switch_cycles;
    logic [COUNTER_W-1:0] perf_pipeline_bubble_cycles;

    string reset_case;
    int injection_cycle;
    int clean_runs;

    paper_interleaved_attention_datapath #(
        .PE_NUM(PE_NUM),
        .D_HEAD(D_HEAD),
        .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .META_W(META_W),
        .COUNTER_W(COUNTER_W)
    ) dut (
        .clk                                      (clk),
        .rst_n                                    (rst_n),
        .load_valid                               (load_valid),
        .load_ready                               (load_ready),
        .load_kind                                (load_kind),
        .load_token                               (load_token),
        .load_dim                                 (load_dim),
        .load_data                                (load_data),
        .start_valid                              (start_valid),
        .start_ready                              (start_ready),
        .start_seq_len                            (start_seq_len),
        .start_meta                               (start_meta),
        .output_valid                             (output_valid),
        .output_ready                             (output_ready),
        .output_base_dim                          (output_base_dim),
        .output_vector_fp32                       (output_vector_fp32),
        .output_lane_mask                         (output_lane_mask),
        .output_status                            (output_status),
        .output_invalid                           (output_invalid),
        .output_meta                              (output_meta),
        .output_last                              (output_last),
        .done_valid                               (done_valid),
        .done_ready                               (done_ready),
        .done_status                              (done_status),
        .done_invalid                             (done_invalid),
        .done_meta                                (done_meta),
        .perf_total_attention_cycles              (perf_total_attention_cycles),
        .perf_qk_cycles                           (perf_qk_cycles),
        .perf_qk_pe_busy_cycles                   (perf_qk_pe_busy_cycles),
        .perf_scale_cycles                        (perf_scale_cycles),
        .perf_reduction_cycles                    (perf_reduction_cycles),
        .perf_reduction_finalize_cycles           (perf_reduction_finalize_cycles),
        .perf_normalization_cycles                (perf_normalization_cycles),
        .perf_sv_cycles                           (perf_sv_cycles),
        .perf_pe_stall_cycles                     (perf_pe_stall_cycles),
        .perf_sfu_stall_cycles                    (perf_sfu_stall_cycles),
        .perf_buffer_stall_cycles                 (perf_buffer_stall_cycles),
        .perf_output_stall_cycles                 (perf_output_stall_cycles),
        .perf_score_buffer_peak_occupancy         (perf_score_buffer_peak_occupancy),
        .perf_paper_array_active_cycles           (perf_paper_array_active_cycles),
        .perf_paper_array_idle_cycles             (perf_paper_array_idle_cycles),
        .perf_inner_mode_cycles                   (perf_inner_mode_cycles),
        .perf_outer_mode_cycles                   (perf_outer_mode_cycles),
        .perf_group0_active_cycles                (perf_group0_active_cycles),
        .perf_group1_active_cycles                (perf_group1_active_cycles),
        .perf_tail_masked_pe_cycles               (perf_tail_masked_pe_cycles),
        .perf_mode_switch_cycles                  (perf_mode_switch_cycles),
        .perf_array_input_stall_cycles            (perf_array_input_stall_cycles),
        .perf_array_output_stall_cycles           (perf_array_output_stall_cycles),
        .perf_qk_sfu_overlap_cycles               (perf_qk_sfu_overlap_cycles),
        .perf_qk_only_cycles                      (perf_qk_only_cycles),
        .perf_sfu_during_qk_cycles                (perf_sfu_during_qk_cycles),
        .perf_score_fifo_full_stall_cycles        (perf_score_fifo_full_stall_cycles),
        .perf_score_fifo_empty_cycles             (perf_score_fifo_empty_cycles),
        .perf_score_fifo_peak_occupancy           (perf_score_fifo_peak_occupancy),
        .perf_sfu_sv_overlap_cycles               (perf_sfu_sv_overlap_cycles),
        .perf_sfu_only_cycles                     (perf_sfu_only_cycles),
        .perf_sv_only_cycles                      (perf_sv_only_cycles),
        .perf_probability_fifo_full_stall_cycles  (perf_probability_fifo_full_stall_cycles),
        .perf_probability_fifo_empty_stall_cycles (perf_probability_fifo_empty_stall_cycles),
        .perf_probability_fifo_peak_occupancy     (perf_probability_fifo_peak_occupancy),
        .perf_inner_to_outer_switch_cycles        (perf_inner_to_outer_switch_cycles),
        .perf_pipeline_bubble_cycles              (perf_pipeline_bubble_cycles)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    task automatic fail(input string msg);
        begin
            $display("HW_H9_RESET_MATRIX_FAIL case=%s d_head=%0d seq=%0d: %s",
                     reset_case, D_HEAD, `HW_H9_SEQ_LEN, msg);
            $fatal(1);
        end
    endtask

    task automatic apply_reset;
        begin
            rst_n = 1'b0;
            load_valid = 1'b0;
            start_valid = 1'b0;
            output_ready = 1'b0;
            done_ready = 1'b0;
            repeat (6) @(posedge clk);
            rst_n = 1'b1;
            repeat (3) @(posedge clk);
        end
    endtask

    task automatic check_reset_state;
        begin
            if (output_valid || done_valid) fail("ghost output/done valid after reset");
            if (!load_ready || !start_ready) fail("load/start ready not restored after reset");
            if (dut.phase_q !== 0) fail("phase did not return to idle");
            if (dut.qk_state_q !== 0) fail("QK state did not reset");
            if (dut.sv_state_q !== 0) fail("sV state did not reset");
            if (dut.qk_token_q !== '0 || dut.reduce_token_q !== '0 ||
                dut.norm_token_q !== '0 || dut.sv_token_q !== '0) begin
                fail("token indices not cleared");
            end
            if (dut.score_valid_count_q !== '0 || dut.prob_valid_count_q !== '0) begin
                fail("score/probability occupancy not cleared");
            end
            if (dut.active_q || dut.softmax_final_seen_q || dut.array_result_seen_q ||
                dut.array_done_seen_q) begin
                fail("in-flight flags not cleared");
            end
            if (^({output_status, done_status, output_meta, done_meta}) === 1'bx) begin
                fail("X/Z on valid/status/meta signals after reset");
            end
        end
    endtask

    task automatic load_one(input logic [1:0] kind, input int token, input int dim, input logic [15:0] data);
        int wait_cycles;
        begin
            load_kind = kind;
            load_token = token[TOKEN_W-1:0];
            load_dim = dim[D_ADDR_W-1:0];
            load_data = data;
            wait_cycles = 0;
            while (load_ready !== 1'b1) begin
                @(negedge clk);
                wait_cycles++;
                if (wait_cycles > 2000) fail("timeout waiting for load_ready");
            end
            @(negedge clk);
            load_valid = 1'b1;
            @(negedge clk);
            load_valid = 1'b0;
        end
    endtask

    task automatic load_dataset;
        begin
            for (int dim = 0; dim < D_HEAD; dim++) begin
                load_one(2'd0, 0, dim, 16'h0000);
            end
            for (int token = 0; token < `HW_H9_SEQ_LEN; token++) begin
                for (int dim = 0; dim < D_HEAD; dim++) begin
                    load_one(2'd1, token, dim, 16'h0000);
                    load_one(2'd2, token, dim, 16'h3C00);
                end
            end
        end
    endtask

    function automatic bit load_case(input string name);
        begin
            load_case = (name == "idle") ||
                        (name == "token_input_first_dimension") ||
                        (name == "token_input_middle_dimension") ||
                        (name == "token_input_last_dimension") ||
                        (name == "q_projection_active") ||
                        (name == "k_projection_active") ||
                        (name == "v_projection_active") ||
                        (name == "qkv_quantization") ||
                        (name == "qkv_stream_to_attention");
        end
    endfunction

    function automatic bit reached_case(input string name);
        begin
            reached_case = 1'b0;
            if (name == "qk_command_issue") reached_case = dut.array_cmd_valid && (dut.phase_q == 1) && (dut.qk_state_q == 1);
            else if (name == "qk_command_accepted") reached_case = dut.array_cmd_fire && (dut.phase_q == 1);
            else if (name == "qk_operation_in_flight") reached_case = (dut.phase_q == 1) && (dut.qk_state_q == 2);
            else if (name == "qk_result_waiting") reached_case = dut.array_result_valid && (dut.phase_q == 1);
            else if (name == "score_packet_valid_but_stalled") reached_case = dut.scaler_out_valid;
            else if (name == "score_fifo_empty") reached_case = (dut.phase_q == 1) && (dut.score_fifo_occupancy == '0);
            else if (name == "score_fifo_occupancy_1") reached_case = (dut.score_fifo_occupancy == 1);
            else if (name == "score_fifo_near_full") reached_case = (dut.score_fifo_occupancy >= 1);
            else if (name == "score_fifo_producer_stalled") reached_case = dut.scaler_out_valid;
            else if (name == "score_fifo_consumer_stalled") reached_case = dut.reduction_in_valid;
            else if (name == "first_score_accepted") reached_case = dut.reduction_in_fire && (dut.reduce_token_q == '0);
            else if (name == "running_max_update") reached_case = dut.reduction_busy;
            else if (name == "exp_issue") reached_case = dut.reduction_in_fire;
            else if (name == "exp_in_flight") reached_case = dut.reduction_busy;
            else if (name == "exp_result_waiting") reached_case = dut.reduction_final_valid;
            else if (name == "exp_sum_update") reached_case = dut.reduction_in_fire && (dut.reduce_token_q != '0);
            else if (name == "score_replay_start") reached_case = (dut.phase_q == 3) && (dut.norm_token_q == '0);
            else if (name == "score_replay_middle") reached_case = (dut.phase_q == 3) && (dut.norm_token_q > 0);
            else if (name == "reciprocal_issue") reached_case = (dut.phase_q == 2);
            else if (name == "reciprocal_in_flight") reached_case = dut.norm_busy;
            else if (name == "normalization") reached_case = dut.norm_score_valid || dut.norm_prob_valid;
            else if (name == "normalized_probability_waiting") reached_case = dut.norm_prob_valid;
            else if (name == "probability_fifo_occupancy_1") reached_case = (dut.prob_fifo_occupancy == 1);
            else if (name == "probability_fifo_near_full") reached_case = (dut.prob_fifo_occupancy >= 1);
            else if (name == "probability_producer_stalled") reached_case = dut.norm_prob_valid;
            else if (name == "probability_consumer_stalled") reached_case = (dut.phase_q == 3) && (dut.prob_fifo_occupancy == '0);
            else if (name == "inner_drain") reached_case = (dut.phase_q == 1) && (dut.qk_state_q == 5);
            else if (name == "mode_switch_request") reached_case = (dut.phase_q == 2);
            else if (name == "mode_switch_wait") reached_case = (dut.phase_q == 2) && dut.norm_start_valid;
            else if (name == "outer_mode_entry") reached_case = (dut.phase_q == 3) && (dut.sv_state_q == 1);
            else if (name == "sv_command_issue") reached_case = dut.array_cmd_valid && (dut.phase_q == 3);
            else if (name == "sv_command_accepted") reached_case = dut.array_cmd_fire && (dut.phase_q == 3);
            else if (name == "sv_operation_in_flight") reached_case = (dut.phase_q == 3) && (dut.sv_state_q == 2);
            else if (name == "sv_result_waiting") reached_case = dut.array_result_valid && (dut.phase_q == 3);
            else if (name == "outer_drain") reached_case = (dut.phase_q == 3) && (dut.sv_state_q == 3);
            else if (name == "head_concat") reached_case = (dut.phase_q == 4);
            else if (name == "head_output_stall") reached_case = output_valid;
            else if (name == "layer_done_stall") reached_case = done_valid;
            else reached_case = (dut.phase_q != 0);
        end
    endfunction

    task automatic wait_for_injection;
        int cycles;
        begin
            cycles = 0;
            while (!reached_case(reset_case)) begin
                @(posedge clk);
                cycles++;
                if (cycles > 200000) fail("timeout waiting for reset injection point");
            end
            injection_cycle = cycles;
        end
    endtask

    task automatic start_transaction(input logic [META_W-1:0] meta);
        int wait_cycles;
        begin
            start_seq_len = SEQ_LEN_W'(`HW_H9_SEQ_LEN);
            start_meta = meta;
            wait_cycles = 0;
            while (start_ready !== 1'b1) begin
                @(negedge clk);
                wait_cycles++;
                if (wait_cycles > 2000) fail("timeout waiting for start_ready");
            end
            @(negedge clk);
            start_valid = 1'b1;
            @(negedge clk);
            start_valid = 1'b0;
        end
    endtask

    task automatic run_clean_transaction(input logic [META_W-1:0] meta);
        int outputs;
        int cycles;
        bit done_seen;
        begin
            output_ready = 1'b1;
            done_ready = 1'b1;
            load_dataset();
            start_transaction(meta);
            outputs = 0;
            cycles = 0;
            done_seen = 1'b0;
            while (!done_seen) begin
                @(posedge clk);
                if (output_valid && output_ready) begin
                    outputs++;
                    if (output_invalid) fail("clean output invalid");
                    if (output_meta !== meta) fail("clean output metadata mismatch");
                    for (int lane = 0; lane < PE_NUM; lane++) begin
                        if (output_lane_mask[lane] &&
                            output_vector_fp32[lane*32 +: 32] !== 32'h3F80_0000) begin
                            fail("clean output lane not FP32 one");
                        end
                    end
                end
                if (done_valid && done_ready) begin
                    if (done_invalid) fail("clean done invalid");
                    if (done_meta !== meta) fail("clean done metadata mismatch");
                    if (outputs != EXPECTED_OUTPUTS) fail("clean output count mismatch");
                    done_seen = 1'b1;
                end
                cycles++;
                if (cycles > 500000) fail("clean recovery transaction timeout");
            end
            clean_runs++;
            output_ready = 1'b0;
            done_ready = 1'b0;
            repeat (2) @(posedge clk);
        end
    endtask

    initial begin
        if (!$value$plusargs("RESET_CASE=%s", reset_case)) begin
            reset_case = "idle";
        end
        rst_n = 1'b0;
        load_valid = 1'b0;
        start_valid = 1'b0;
        start_seq_len = SEQ_LEN_W'(`HW_H9_SEQ_LEN);
        start_meta = 16'h9000;
        output_ready = 1'b0;
        done_ready = 1'b0;
        clean_runs = 0;
        injection_cycle = -1;
        apply_reset();

        if (load_case(reset_case)) begin
            if (reset_case == "idle") begin
                injection_cycle = 0;
            end else begin
                load_one(2'd0, 0, 0, 16'h0000);
                injection_cycle = 1;
            end
        end else begin
            load_dataset();
            start_transaction(16'h9001);
            if (reset_case == "head_output_stall") output_ready = 1'b0;
            else output_ready = 1'b1;
            if (reset_case == "layer_done_stall") done_ready = 1'b0;
            else done_ready = 1'b1;
            wait_for_injection();
        end

        apply_reset();
        check_reset_state();
        run_clean_transaction(16'hA001);
        run_clean_transaction(16'hA002);
        $display("HW_H9_RESET_MATRIX_PASS case=%s config=H1/D%0d seq=%0d injection_cycle=%0d score_occ_after=0 prob_occ_after=0 clean_runs=%0d qk_sfu_overlap=%0d sfu_sv_overlap=%0d score_peak=%0d prob_peak=%0d",
                 reset_case, D_HEAD, `HW_H9_SEQ_LEN, injection_cycle, clean_runs,
                 perf_qk_sfu_overlap_cycles, perf_sfu_sv_overlap_cycles,
                 perf_score_fifo_peak_occupancy, perf_probability_fifo_peak_occupancy);
        $finish;
    end
endmodule

`default_nettype wire
