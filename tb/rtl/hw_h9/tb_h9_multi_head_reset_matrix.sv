`timescale 1ns/1ps
`default_nettype none

`ifndef STAGE5_N_HEAD
`define STAGE5_N_HEAD 2
`endif

`ifndef STAGE5_D_HEAD
`define STAGE5_D_HEAD 8
`endif

`ifndef STAGE5_MAX_SEQ_LEN
`define STAGE5_MAX_SEQ_LEN 8
`endif

module tb_h9_multi_head_reset_matrix;
    localparam int N_HEAD = `STAGE5_N_HEAD;
    localparam int D_HEAD = `STAGE5_D_HEAD;
    localparam int PE_NUM = 8;
    localparam int MAX_SEQ_LEN = `STAGE5_MAX_SEQ_LEN;
    localparam int META_W = 16;
    localparam int COUNTER_W = 64;
    localparam int HEAD_W = (N_HEAD <= 1) ? 1 : $clog2(N_HEAD);
    localparam int DIM_W = (D_HEAD <= 1) ? 1 : $clog2(D_HEAD);
    localparam int SEQ_LEN_W = (MAX_SEQ_LEN <= 1) ? 1 : $clog2(MAX_SEQ_LEN + 1);
    localparam int TILES = (D_HEAD + PE_NUM - 1) / PE_NUM;

    localparam logic [31:0] FP32_ONE = 32'h3f800000;

    logic clk;
    logic rst_n;
    logic token_valid;
    logic token_ready;
    logic [HEAD_W-1:0] token_head;
    logic [DIM_W-1:0] token_dim;
    logic [15:0] token_q_fp16;
    logic [15:0] token_k_fp16;
    logic [15:0] token_v_fp16;
    logic token_last_dim;
    logic token_last_head;
    logic [META_W-1:0] token_meta;
    logic output_valid;
    logic output_ready;
    logic [HEAD_W-1:0] output_head;
    logic [DIM_W-1:0] output_base_dim;
    logic [PE_NUM*32-1:0] output_vector_fp32;
    logic [PE_NUM-1:0] output_lane_mask;
    logic [7:0] output_status;
    logic output_invalid;
    logic [META_W-1:0] output_meta;
    logic output_last_tile;
    logic output_last_head;
    logic output_last_token;
    logic done_valid;
    logic done_ready;
    logic [7:0] done_status;
    logic done_invalid;
    logic [META_W-1:0] done_meta;
    logic [SEQ_LEN_W-1:0] done_valid_seq_len;
    logic [SEQ_LEN_W-1:0] current_valid_seq_len;

    logic [COUNTER_W-1:0] perf_generation_steps;
    logic [COUNTER_W-1:0] perf_total_cycles;
    logic [COUNTER_W-1:0] perf_per_head_attention_cycles;
    logic [COUNTER_W-1:0] perf_head_switch_cycles;
    logic [COUNTER_W-1:0] perf_provisional_write_cycles;
    logic [COUNTER_W-1:0] perf_cache_read_cycles;
    logic [COUNTER_W-1:0] perf_cache_write_cycles;
    logic [COUNTER_W-1:0] perf_cache_stall_cycles;
    logic [COUNTER_W-1:0] perf_commit_cycles;
    logic [COUNTER_W-1:0] perf_pe_stall_cycles;
    logic [COUNTER_W-1:0] perf_sfu_stall_cycles;
    logic [COUNTER_W-1:0] perf_output_stall_cycles;
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
    logic [SEQ_LEN_W-1:0] perf_peak_valid_seq_len;

    string reset_case;
    int injection_cycle;
    int recovery_commits;
    int recovery_outputs;
    int recovery_done;

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    multi_head_generation_engine #(
        .N_HEAD(N_HEAD),
        .PE_NUM(PE_NUM),
        .D_HEAD(D_HEAD),
        .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .META_W(META_W),
        .COUNTER_W(COUNTER_W),
        .ATTENTION_PE_ARCH(1),
        .ATTENTION_SCHEDULE(1)
    ) u_dut (
        .clk                            (clk),
        .rst_n                          (rst_n),
        .token_valid                    (token_valid),
        .token_ready                    (token_ready),
        .token_head                     (token_head),
        .token_dim                      (token_dim),
        .token_q_fp16                   (token_q_fp16),
        .token_k_fp16                   (token_k_fp16),
        .token_v_fp16                   (token_v_fp16),
        .token_last_dim                 (token_last_dim),
        .token_last_head                (token_last_head),
        .token_meta                     (token_meta),
        .output_valid                   (output_valid),
        .output_ready                   (output_ready),
        .output_head                    (output_head),
        .output_base_dim                (output_base_dim),
        .output_vector_fp32             (output_vector_fp32),
        .output_lane_mask               (output_lane_mask),
        .output_status                  (output_status),
        .output_invalid                 (output_invalid),
        .output_meta                    (output_meta),
        .output_last_tile               (output_last_tile),
        .output_last_head               (output_last_head),
        .output_last_token              (output_last_token),
        .done_valid                     (done_valid),
        .done_ready                     (done_ready),
        .done_status                    (done_status),
        .done_invalid                   (done_invalid),
        .done_meta                      (done_meta),
        .done_valid_seq_len             (done_valid_seq_len),
        .current_valid_seq_len          (current_valid_seq_len),
        .perf_generation_steps          (perf_generation_steps),
        .perf_total_cycles              (perf_total_cycles),
        .perf_per_head_attention_cycles (perf_per_head_attention_cycles),
        .perf_head_switch_cycles        (perf_head_switch_cycles),
        .perf_provisional_write_cycles  (perf_provisional_write_cycles),
        .perf_cache_read_cycles         (perf_cache_read_cycles),
        .perf_cache_write_cycles        (perf_cache_write_cycles),
        .perf_cache_stall_cycles        (perf_cache_stall_cycles),
        .perf_commit_cycles             (perf_commit_cycles),
        .perf_pe_stall_cycles           (perf_pe_stall_cycles),
        .perf_sfu_stall_cycles          (perf_sfu_stall_cycles),
        .perf_output_stall_cycles       (perf_output_stall_cycles),
        .perf_paper_array_active_cycles (perf_paper_array_active_cycles),
        .perf_paper_array_idle_cycles   (perf_paper_array_idle_cycles),
        .perf_inner_mode_cycles         (perf_inner_mode_cycles),
        .perf_outer_mode_cycles         (perf_outer_mode_cycles),
        .perf_group0_active_cycles      (perf_group0_active_cycles),
        .perf_group1_active_cycles      (perf_group1_active_cycles),
        .perf_tail_masked_pe_cycles     (perf_tail_masked_pe_cycles),
        .perf_mode_switch_cycles        (perf_mode_switch_cycles),
        .perf_array_input_stall_cycles  (perf_array_input_stall_cycles),
        .perf_array_output_stall_cycles (perf_array_output_stall_cycles),
        .perf_peak_valid_seq_len        (perf_peak_valid_seq_len)
    );

    wire [3:0] mh_state = u_dut.u_multi_head_generation_controller.state_q;
    wire [HEAD_W-1:0] mh_head = u_dut.u_multi_head_generation_controller.current_head_q;
    wire [N_HEAD-1:0] mh_done_seen = u_dut.u_multi_head_generation_controller.head_done_seen_q;
    wire mh_cache_commit_valid = u_dut.cache_commit_valid;
    wire mh_cache_commit_ready = u_dut.cache_commit_ready;
    wire sha_output_stalled = u_dut.sha_output_valid && !u_dut.sha_output_ready;
    wire top_output_stalled = output_valid && !output_ready;
    wire top_done_stalled = done_valid && !done_ready;
    wire [2:0] dp_phase =
        u_dut.u_shared_single_head_attention.g_interleaved_schedule.u_interleaved_datapath.phase_q;
    wire [2:0] dp_qk_state =
        u_dut.u_shared_single_head_attention.g_interleaved_schedule.u_interleaved_datapath.qk_state_q;
    wire [1:0] dp_sv_state =
        u_dut.u_shared_single_head_attention.g_interleaved_schedule.u_interleaved_datapath.sv_state_q;
    wire [SEQ_LEN_W-1:0] dp_score_occupancy =
        u_dut.u_shared_single_head_attention.g_interleaved_schedule.u_interleaved_datapath.score_fifo_occupancy;
    wire [SEQ_LEN_W-1:0] dp_prob_occupancy =
        u_dut.u_shared_single_head_attention.g_interleaved_schedule.u_interleaved_datapath.prob_fifo_occupancy;
    wire dp_reduction_active =
        u_dut.u_shared_single_head_attention.g_interleaved_schedule.u_interleaved_datapath.reduction_busy ||
        u_dut.u_shared_single_head_attention.g_interleaved_schedule.u_interleaved_datapath.reduction_in_valid ||
        u_dut.u_shared_single_head_attention.g_interleaved_schedule.u_interleaved_datapath.reduction_final_valid;
    wire dp_norm_active =
        u_dut.u_shared_single_head_attention.g_interleaved_schedule.u_interleaved_datapath.norm_busy ||
        u_dut.u_shared_single_head_attention.g_interleaved_schedule.u_interleaved_datapath.norm_score_valid ||
        u_dut.u_shared_single_head_attention.g_interleaved_schedule.u_interleaved_datapath.norm_prob_valid;

    task automatic fail(input string msg);
        begin
            $display("HW_H9_MULTI_HEAD_RESET_FAIL case=%s config=H%0d/D%0d: %s",
                     reset_case, N_HEAD, D_HEAD, msg);
            $fatal(1);
        end
    endtask

    task automatic apply_reset;
        begin
            rst_n = 1'b0;
            token_valid = 1'b0;
            token_head = '0;
            token_dim = '0;
            token_q_fp16 = 16'h0000;
            token_k_fp16 = 16'h0000;
            token_v_fp16 = 16'h3c00;
            token_last_dim = 1'b0;
            token_last_head = 1'b0;
            token_meta = '0;
            output_ready = 1'b0;
            done_ready = 1'b0;
            repeat (8) @(posedge clk);
            rst_n = 1'b1;
            repeat (3) @(posedge clk);
        end
    endtask

    task automatic check_reset_state;
        begin
            #1;
            if (!token_ready) fail("token_ready not restored after reset");
            if (output_valid || done_valid) fail("ghost output/done valid after reset");
            if (current_valid_seq_len !== '0) fail("valid_seq_len nonzero after reset");
            if (mh_state !== 4'd0) fail("multi-head controller not in ST_LOAD_TOKEN after reset");
            if (mh_head !== '0) fail("current head not cleared after reset");
            if (mh_done_seen !== '0) fail("head_done_seen not cleared after reset");
            if (dp_phase !== 3'd0) fail("single-head datapath phase not idle after reset");
            if (dp_score_occupancy !== '0) fail("score FIFO occupancy not cleared after reset");
            if (dp_prob_occupancy !== '0) fail("probability FIFO occupancy not cleared after reset");
            if (^({output_valid, done_valid, output_status, done_status, done_valid_seq_len}) === 1'bx) begin
                fail("X/Z on valid/status signals after reset");
            end
        end
    endtask

    task automatic drive_dim(input int head, input int dim, input logic [META_W-1:0] meta);
        int cycles;
        logic pre_fire;
        begin
            cycles = 0;
            @(negedge clk);
            token_valid = 1'b1;
            token_head = HEAD_W'(head);
            token_dim = DIM_W'(dim);
            token_q_fp16 = 16'h0000;
            token_k_fp16 = 16'h0000;
            token_v_fp16 = 16'h3c00;
            token_last_dim = (dim == D_HEAD - 1);
            token_last_head = (head == N_HEAD - 1) && (dim == D_HEAD - 1);
            token_meta = meta;
            do begin
                #1;
                pre_fire = token_valid && token_ready;
                @(posedge clk); #1;
                cycles++;
                if (cycles > 2000) fail("timeout while driving token dimension");
                if (!pre_fire) @(negedge clk);
            end while (!pre_fire);
            @(negedge clk);
            token_valid = 1'b0;
            token_last_dim = 1'b0;
            token_last_head = 1'b0;
        end
    endtask

    task automatic drive_token(input logic [META_W-1:0] meta);
        begin
            for (int head = 0; head < N_HEAD; head++) begin
                for (int dim = 0; dim < D_HEAD; dim++) begin
                    drive_dim(head, dim, meta);
                    if (((head * D_HEAD + dim) % 5) == 2) begin
                        @(posedge clk);
                    end
                end
            end
        end
    endtask

    function automatic bit is_middle_head;
        begin
            is_middle_head = (N_HEAD > 2) && (mh_head > '0) && (mh_head < HEAD_W'(N_HEAD - 1));
        end
    endfunction

    function automatic bit reached_case(input string name);
        begin
            reached_case = 1'b0;
            if (name == "first_head_qk_active") begin
                reached_case = (mh_head == '0) && (dp_phase == 3'd1) && (dp_qk_state inside {3'd1, 3'd2, 3'd3, 3'd4});
            end else if (name == "first_head_score_fifo_nonempty") begin
                reached_case = (mh_head == '0) && (dp_score_occupancy > 0);
            end else if (name == "first_head_sfu_active") begin
                reached_case = (mh_head == '0) && dp_reduction_active;
            end else if (name == "first_head_probability_fifo_nonempty") begin
                reached_case = (mh_head == '0) && (dp_prob_occupancy > 0);
            end else if (name == "first_head_sv_active") begin
                reached_case = (mh_head == '0) && (dp_phase == 3'd3) && (dp_sv_state inside {2'd1, 2'd2});
            end else if (name == "first_head_result_waiting") begin
                reached_case = (mh_head == '0) && sha_output_stalled;
            end else if (name == "first_head_done_next_not_started") begin
                reached_case = (mh_head == '0) && (mh_state == 4'd9) && mh_done_seen[0];
            end else if (name == "head_boundary") begin
                reached_case = (mh_state == 4'd9);
            end else if (name == "middle_head_qk_active") begin
                reached_case = is_middle_head() && (dp_phase == 3'd1) && (dp_qk_state inside {3'd1, 3'd2, 3'd3, 3'd4});
            end else if (name == "middle_head_sfu_active") begin
                reached_case = is_middle_head() && dp_reduction_active;
            end else if (name == "middle_head_probability_fifo_nonempty") begin
                reached_case = is_middle_head() && (dp_prob_occupancy > 0);
            end else if (name == "middle_head_sv_active") begin
                reached_case = is_middle_head() && (dp_phase == 3'd3) && (dp_sv_state inside {2'd1, 2'd2});
            end else if (name == "final_head_active") begin
                reached_case = (mh_head == HEAD_W'(N_HEAD - 1)) && (mh_state == 4'd8);
            end else if (name == "all_heads_computed_before_commit") begin
                reached_case = (mh_state == 4'd10) && (mh_done_seen == {N_HEAD{1'b1}});
            end else if (name == "atomic_kv_commit_cycle") begin
                reached_case = mh_cache_commit_valid && mh_cache_commit_ready;
            end else if (name == "commit_after_before_output") begin
                reached_case = (mh_state == 4'd10) && mh_cache_commit_valid && !output_valid;
            end else if (name == "multi_head_output_stalled") begin
                reached_case = top_output_stalled;
            end else if (name == "multi_head_done_stalled") begin
                reached_case = top_done_stalled;
            end
        end
    endfunction

    task automatic configure_stall_for_case(input string name);
        begin
            output_ready = 1'b1;
            done_ready = 1'b1;
            if ((name == "first_head_result_waiting") || (name == "multi_head_output_stalled")) begin
                output_ready = 1'b0;
            end
            if (name == "multi_head_done_stalled") begin
                done_ready = 1'b0;
            end
        end
    endtask

    task automatic wait_for_injection(input string name);
        int cycles;
        begin
            configure_stall_for_case(name);
            drive_token(16'h9100);
            cycles = 0;
            while (!reached_case(name)) begin
                @(posedge clk);
                cycles++;
                if (cycles > 500000) begin
                    fail({"timeout waiting for real injection point ", name});
                end
            end
            injection_cycle = cycles;
        end
    endtask

    task automatic run_clean_token(input int expected_seq_after, input logic [META_W-1:0] meta);
        int cycles;
        int out_tiles;
        bit done_seen;
        logic [31:0] lane_value;
        begin
            output_ready = 1'b1;
            done_ready = 1'b1;
            drive_token(meta);
            cycles = 0;
            out_tiles = 0;
            done_seen = 1'b0;
            while (!done_seen) begin
                @(posedge clk);
                #1;
                if (output_valid && output_ready) begin
                    if (output_invalid) fail("clean recovery output invalid");
                    if (output_meta !== meta) fail("clean recovery output metadata mismatch");
                    for (int lane = 0; lane < PE_NUM; lane++) begin
                        if (output_lane_mask[lane]) begin
                            lane_value = output_vector_fp32[lane*32 +: 32];
                            if (lane_value !== FP32_ONE) begin
                                $display("CHECK_FAIL clean output head=%0d base=%0d lane=%0d got=%08h expected=%08h",
                                         output_head, output_base_dim, lane, lane_value, FP32_ONE);
                                $fatal(1);
                            end
                        end
                    end
                    out_tiles++;
                    recovery_outputs++;
                end
                if (done_valid && done_ready) begin
                    if (done_invalid) fail("clean recovery done invalid");
                    if (done_meta !== meta) fail("clean recovery done metadata mismatch");
                    if (done_valid_seq_len !== SEQ_LEN_W'(expected_seq_after)) fail("clean recovery seq len mismatch");
                    done_seen = 1'b1;
                    recovery_commits++;
                    recovery_done++;
                end
                cycles++;
                if (cycles > (N_HEAD * D_HEAD * expected_seq_after * 500 + 200000)) begin
                    fail("clean recovery timeout");
                end
            end
            if (out_tiles != (N_HEAD * TILES)) fail("clean recovery output tile count mismatch");
        end
    endtask

    initial begin
        if (!$value$plusargs("RESET_CASE=%s", reset_case)) begin
            reset_case = "first_head_qk_active";
        end
        recovery_commits = 0;
        recovery_outputs = 0;
        recovery_done = 0;
        apply_reset();
        wait_for_injection(reset_case);
        apply_reset();
        check_reset_state();
        run_clean_token(1, 16'h9201);
        run_clean_token(2, 16'h9202);
        $display("HW_H9_MULTI_HEAD_RESET_PASS case=%s config=H%0d/D%0d injection_cycle=%0d recovery_commits=%0d recovery_outputs=%0d recovery_done=%0d trigger=independent_real_hierarchy",
                 reset_case, N_HEAD, D_HEAD, injection_cycle, recovery_commits,
                 recovery_outputs, recovery_done);
        $finish;
    end
endmodule

`default_nettype wire
