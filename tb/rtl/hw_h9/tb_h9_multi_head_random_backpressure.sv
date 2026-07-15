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

module tb_h9_multi_head_random_backpressure;
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
    localparam logic [31:0] FP32_EXPECTED = 32'h00000000;

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

    int seed;
    int token_count;
    int pattern;
    int cycles;
    int watchdog_limit;
    int source_gap_cycles;
    int output_stall_cycles;
    int done_stall_cycles;
    int simultaneous_stall_cycles;
    int head_boundary_stall_cycles;
    int commit_near_stall_cycles;
    int output_count;
    int done_count;
    int commit_count;
    int score_peak;
    int prob_peak;

    logic [PE_NUM*32-1:0] held_output_vector;
    logic [PE_NUM-1:0] held_output_mask;
    logic [HEAD_W-1:0] held_output_head;
    logic [DIM_W-1:0] held_output_base;
    logic [META_W-1:0] held_output_meta;
    logic held_output_valid;
    logic [META_W-1:0] held_done_meta;
    logic [SEQ_LEN_W-1:0] held_done_seq;
    logic held_done_valid;

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
    wire mh_commit_fire = u_dut.cache_commit_valid && u_dut.cache_commit_ready;
    wire [SEQ_LEN_W-1:0] dp_score_occupancy =
        u_dut.u_shared_single_head_attention.g_interleaved_schedule.u_interleaved_datapath.score_fifo_occupancy;
    wire [SEQ_LEN_W-1:0] dp_prob_occupancy =
        u_dut.u_shared_single_head_attention.g_interleaved_schedule.u_interleaved_datapath.prob_fifo_occupancy;

    task automatic fail(input string msg);
        begin
            $display("HW_H9_MULTI_HEAD_RANDOM_FAIL seed=%0d config=H%0d/D%0d seq=%0d: %s",
                     seed, N_HEAD, D_HEAD, token_count, msg);
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
            token_v_fp16 = 16'h0000;
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

    function automatic bit ready_by_pattern(input int endpoint, input int local_cycle);
        int r;
        begin
            r = $urandom(seed ^ (endpoint * 32'h45d9f3b) ^ local_cycle);
            ready_by_pattern = 1'b1;
            unique case (pattern)
                0: ready_by_pattern = 1'b1;
                1: ready_by_pattern = (local_cycle[0] == 1'b0);
                2: ready_by_pattern = ((local_cycle % 7) != endpoint);
                3: ready_by_pattern = ((local_cycle % 23) < (12 + endpoint));
                4: ready_by_pattern = ((r % 100) >= (25 + endpoint * 8));
                default: ready_by_pattern = ((r % 100) >= (10 + endpoint * 5));
            endcase
        end
    endfunction

    task automatic drive_dim_random(input int head, input int dim, input logic [META_W-1:0] meta);
        int local_cycle;
        logic pre_fire;
        begin
            local_cycle = 0;
            @(negedge clk);
            token_valid = 1'b0;
            while (!ready_by_pattern(0, cycles + local_cycle)) begin
                source_gap_cycles++;
                @(posedge clk);
                local_cycle++;
            end
            @(negedge clk);
            token_valid = 1'b1;
            token_head = HEAD_W'(head);
            token_dim = DIM_W'(dim);
            token_q_fp16 = 16'h0000;
            token_k_fp16 = 16'h0000;
            token_v_fp16 = 16'h0000;
            token_last_dim = (dim == D_HEAD - 1);
            token_last_head = (head == N_HEAD - 1) && (dim == D_HEAD - 1);
            token_meta = meta;
            do begin
                #1;
                pre_fire = token_valid && token_ready;
                @(posedge clk); #1;
                cycles++;
                if (cycles > watchdog_limit) fail("watchdog while driving token dimension");
                if (!pre_fire) @(negedge clk);
            end while (!pre_fire);
            @(negedge clk);
            token_valid = 1'b0;
        end
    endtask

    task automatic drive_token_random(input logic [META_W-1:0] meta);
        begin
            for (int head = 0; head < N_HEAD; head++) begin
                for (int dim = 0; dim < D_HEAD; dim++) begin
                    drive_dim_random(head, dim, meta);
                end
            end
        end
    endtask

    task automatic update_random_ready;
        begin
            output_ready = ready_by_pattern(1, cycles);
            done_ready = ready_by_pattern(2, cycles);
            if (mh_state == 4'd9 && ((cycles % 5) != 0)) begin
                output_ready = 1'b0;
                head_boundary_stall_cycles++;
            end
            if (mh_state == 4'd10 && ((cycles % 7) != 0)) begin
                done_ready = 1'b0;
                commit_near_stall_cycles++;
            end
            if (!output_ready && !done_ready) simultaneous_stall_cycles++;
            if (output_valid && !output_ready) output_stall_cycles++;
            if (done_valid && !done_ready) done_stall_cycles++;
        end
    endtask

    task automatic check_stability;
        begin
            if (held_output_valid) begin
                if (!output_valid) fail("output valid dropped while stalled");
                if ({output_head, output_base_dim, output_vector_fp32, output_lane_mask, output_meta} !==
                    {held_output_head, held_output_base, held_output_vector, held_output_mask, held_output_meta}) begin
                    fail("output payload changed while stalled");
                end
            end
            if (held_done_valid) begin
                if (!done_valid) fail("done valid dropped while stalled");
                if ({done_meta, done_valid_seq_len} !== {held_done_meta, held_done_seq}) begin
                    fail("done payload changed while stalled");
                end
            end
            held_output_valid = output_valid && !output_ready;
            held_output_head = output_head;
            held_output_base = output_base_dim;
            held_output_vector = output_vector_fp32;
            held_output_mask = output_lane_mask;
            held_output_meta = output_meta;
            held_done_valid = done_valid && !done_ready;
            held_done_meta = done_meta;
            held_done_seq = done_valid_seq_len;
        end
    endtask

    task automatic run_one_token(input int token_idx);
        int expected_tiles;
        int tiles_seen;
        bit done_seen;
        logic [META_W-1:0] meta;
        logic [31:0] lane_value;
        begin
            meta = META_W'(16'ha000 + token_idx);
            expected_tiles = N_HEAD * TILES;
            tiles_seen = 0;
            done_seen = 1'b0;
            drive_token_random(meta);
            while (!done_seen) begin
                @(negedge clk);
                update_random_ready();
                #1;
                check_stability();
                if (output_valid && output_ready) begin
                    if (output_invalid) fail("output invalid");
                    if (output_meta !== meta) fail("output metadata mismatch");
                    if (^({output_head, output_base_dim, output_vector_fp32, output_lane_mask}) === 1'bx) begin
                        fail("X/Z in output payload");
                    end
                    for (int lane = 0; lane < PE_NUM; lane++) begin
                        if (output_lane_mask[lane]) begin
                            lane_value = output_vector_fp32[lane*32 +: 32];
                            if (lane_value !== FP32_EXPECTED) fail("output value mismatch");
                        end
                    end
                    tiles_seen++;
                    output_count++;
                end
                if (mh_commit_fire) commit_count++;
                if (int'(dp_score_occupancy) > score_peak) score_peak = int'(dp_score_occupancy);
                if (int'(dp_prob_occupancy) > prob_peak) prob_peak = int'(dp_prob_occupancy);
                if (done_valid && done_ready) begin
                    if (done_invalid) fail("done invalid");
                    if (done_meta !== meta) fail("done metadata mismatch");
                    if (done_valid_seq_len !== SEQ_LEN_W'(token_idx + 1)) fail("done seq mismatch");
                    done_seen = 1'b1;
                    done_count++;
                end
                @(posedge clk);
                cycles++;
                if (cycles > watchdog_limit) fail("watchdog timeout");
            end
            if (tiles_seen != expected_tiles) fail("output tile count mismatch");
        end
    endtask

    initial begin
        if (!$value$plusargs("SEED=%d", seed)) seed = 101;
        if (!$value$plusargs("TOKEN_COUNT=%d", token_count)) token_count = MAX_SEQ_LEN;
        if (!$value$plusargs("PATTERN=%d", pattern)) pattern = 0;
        if (token_count > MAX_SEQ_LEN) token_count = MAX_SEQ_LEN;
        watchdog_limit = (N_HEAD * token_count * (D_HEAD + token_count) * 250) + 50000;
        source_gap_cycles = 0;
        output_stall_cycles = 0;
        done_stall_cycles = 0;
        simultaneous_stall_cycles = 0;
        head_boundary_stall_cycles = 0;
        commit_near_stall_cycles = 0;
        output_count = 0;
        done_count = 0;
        commit_count = 0;
        score_peak = 0;
        prob_peak = 0;
        cycles = 0;
        held_output_valid = 1'b0;
        held_done_valid = 1'b0;
        apply_reset();
        for (int tok = 0; tok < token_count; tok++) begin
            run_one_token(tok);
        end
        if (commit_count != token_count) fail("commit count mismatch");
        if (done_count != token_count) fail("done count mismatch");
        if (output_count != token_count * N_HEAD * TILES) fail("output count mismatch");
        $display("HW_H9_MULTI_HEAD_RANDOM_PASS seed=%0d config=H%0d/D%0d seq=%0d token_count=%0d pattern=%0d cycles=%0d watchdog=%0d source_gap=%0d output_stall=%0d done_stall=%0d simultaneous=%0d head_boundary_stall=%0d commit_near_stall=%0d score_peak=%0d prob_peak=%0d commits=%0d outputs=%0d done=%0d",
                 seed, N_HEAD, D_HEAD, MAX_SEQ_LEN, token_count, pattern, cycles,
                 watchdog_limit, source_gap_cycles, output_stall_cycles,
                 done_stall_cycles, simultaneous_stall_cycles,
                 head_boundary_stall_cycles, commit_near_stall_cycles,
                 score_peak, prob_peak, commit_count, output_count, done_count);
        $finish;
    end

endmodule

`default_nettype wire
