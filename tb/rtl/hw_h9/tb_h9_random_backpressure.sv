`timescale 1ns/1ps
`default_nettype none

`ifndef HW_H9_D_HEAD
`define HW_H9_D_HEAD 8
`endif

`ifndef HW_H9_SEQ_LEN
`define HW_H9_SEQ_LEN 8
`endif

`ifndef HW_H9_TOKEN_COUNT
`define HW_H9_TOKEN_COUNT 2
`endif

module tb_h9_random_backpressure;
    localparam int PE_NUM = 8;
    localparam int D_HEAD = `HW_H9_D_HEAD;
    localparam int MAX_SEQ_LEN = 64;
    localparam int META_W = 16;
    localparam int COUNTER_W = 64;
    localparam int TOKEN_W = (MAX_SEQ_LEN <= 1) ? 1 : $clog2(MAX_SEQ_LEN);
    localparam int SEQ_LEN_W = (MAX_SEQ_LEN <= 1) ? 1 : $clog2(MAX_SEQ_LEN + 1);
    localparam int D_ADDR_W = (D_HEAD <= 1) ? 1 : $clog2(D_HEAD);
    localparam int EXPECTED_OUTPUTS = (D_HEAD + PE_NUM - 1) / PE_NUM;
    localparam int TILE_COUNT = (D_HEAD + PE_NUM - 1) / PE_NUM;
    localparam int CAL_T = (D_HEAD + 7) / 8;
    localparam int CALIBRATED_CYCLES = 65 * `HW_H9_SEQ_LEN + 127 + 2 * CAL_T;
    localparam int LOAD_OPS = D_HEAD + 2 * `HW_H9_SEQ_LEN * D_HEAD;
    localparam int WATCHDOG_LIMIT = CALIBRATED_CYCLES + (LOAD_OPS * 20) +
                                    (`HW_H9_SEQ_LEN * 80) + (TILE_COUNT * 80) + 2000;

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

    int seed;
    int pattern;
    int cycle_counter;
    int output_stall_cycles;
    int done_stall_cycles;
    int source_gap_cycles;
    int total_outputs;
    int total_done;
    logic [PE_NUM*32-1:0] stalled_output_payload;
    logic [META_W-1:0] stalled_done_meta;
    bit output_was_stalled;
    bit done_was_stalled;

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
            $display("HW_H9_RANDOM_BACKPRESSURE_FAIL seed=%0d pattern=%0d d_head=%0d seq=%0d token_count=%0d: %s",
                     seed, pattern, D_HEAD, `HW_H9_SEQ_LEN, `HW_H9_TOKEN_COUNT, msg);
            $fatal(1);
        end
    endtask

    function automatic bit ready_pattern(input int endpoint, input int cycle);
        int random_value;
        begin
            unique case (pattern)
                0: ready_pattern = 1'b1;
                1: ready_pattern = ((cycle + endpoint) % 2) == 0;
                2: ready_pattern = ((cycle + endpoint) % 5) != 0;
                3: ready_pattern = !(((cycle + seed + endpoint) % 37) >= 7 &&
                                      ((cycle + seed + endpoint) % 37) < 14);
                4: ready_pattern = ((cycle + endpoint) % 19) == 0;
                default: begin
                    random_value = $urandom_range(0, 99);
                    ready_pattern = random_value >= (20 + endpoint * 3);
                end
            endcase
        end
    endfunction

    task automatic apply_reset;
        begin
            rst_n = 1'b0;
            load_valid = 1'b0;
            start_valid = 1'b0;
            repeat (8) @(posedge clk);
            rst_n = 1'b1;
            repeat (4) @(posedge clk);
        end
    endtask

    task automatic random_gap(input int endpoint, input int max_gap);
        int gap;
        begin
            gap = $urandom_range(0, max_gap);
            repeat (gap) begin
                source_gap_cycles++;
                @(posedge clk);
            end
        end
    endtask

    task automatic load_one(input logic [1:0] kind, input int token, input int dim, input logic [15:0] data);
        bit fired;
        begin
            random_gap(0, 5);
            fired = 1'b0;
            while (!fired) begin
                @(negedge clk);
                load_kind = kind;
                load_token = token[TOKEN_W-1:0];
                load_dim = dim[D_ADDR_W-1:0];
                load_data = data;
                load_valid = ready_pattern(0, cycle_counter);
                #1;
                fired = load_valid && load_ready;
                @(posedge clk);
                if (!fired) source_gap_cycles++;
            end
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

    task automatic start_transaction(input logic [META_W-1:0] meta);
        bit fired;
        begin
            start_seq_len = SEQ_LEN_W'(`HW_H9_SEQ_LEN);
            start_meta = meta;
            random_gap(1, 8);
            fired = 1'b0;
            while (!fired) begin
                @(negedge clk);
                start_valid = ready_pattern(1, cycle_counter);
                #1;
                fired = start_valid && start_ready;
                @(posedge clk);
                if (!fired) source_gap_cycles++;
            end
            @(negedge clk);
            start_valid = 1'b0;
        end
    endtask

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_counter <= 0;
            output_stall_cycles <= 0;
            done_stall_cycles <= 0;
            output_was_stalled <= 1'b0;
            done_was_stalled <= 1'b0;
            stalled_output_payload <= '0;
            stalled_done_meta <= '0;
        end else begin
            cycle_counter <= cycle_counter + 1;
            output_ready <= ready_pattern(2, cycle_counter);
            done_ready <= ready_pattern(3, cycle_counter);

            if (output_valid && !output_ready) begin
                output_stall_cycles <= output_stall_cycles + 1;
                if (!output_was_stalled) begin
                    stalled_output_payload <= output_vector_fp32;
                    output_was_stalled <= 1'b1;
                end else if (stalled_output_payload !== output_vector_fp32) begin
                    fail("output payload changed while stalled");
                end
            end else begin
                output_was_stalled <= 1'b0;
            end

            if (done_valid && !done_ready) begin
                done_stall_cycles <= done_stall_cycles + 1;
                if (!done_was_stalled) begin
                    stalled_done_meta <= done_meta;
                    done_was_stalled <= 1'b1;
                end else if (stalled_done_meta !== done_meta) begin
                    fail("done payload changed while stalled");
                end
            end else begin
                done_was_stalled <= 1'b0;
            end
        end
    end

    task automatic run_one(input int token_idx);
        int outputs;
        int cycles;
        bit done_seen;
        begin
            load_dataset();
            start_transaction(META_W'(16'hB000 + token_idx));
            outputs = 0;
            cycles = 0;
            done_seen = 1'b0;
            while (!done_seen) begin
                @(posedge clk);
                if (output_valid && output_ready) begin
                    outputs++;
                    total_outputs++;
                    if (output_invalid) fail("output invalid");
                    if (output_meta !== META_W'(16'hB000 + token_idx)) fail("output metadata mismatch");
                    if (^({output_status, output_base_dim, output_lane_mask, output_vector_fp32}) === 1'bx) begin
                        fail("X on valid output payload");
                    end
                    for (int lane = 0; lane < PE_NUM; lane++) begin
                        if (output_lane_mask[lane] &&
                            output_vector_fp32[lane*32 +: 32] !== 32'h3F80_0000) begin
                            fail("output lane not FP32 one");
                        end
                    end
                end
                if (done_valid && done_ready) begin
                    total_done++;
                    if (done_invalid) fail("done invalid");
                    if (done_meta !== META_W'(16'hB000 + token_idx)) fail("done metadata mismatch");
                    if (outputs != EXPECTED_OUTPUTS) fail("output count mismatch");
                    done_seen = 1'b1;
                end
                cycles++;
                if (cycles > WATCHDOG_LIMIT) begin
                    fail("watchdog timeout");
                end
            end
        end
    endtask

    initial begin
        if (!$value$plusargs("SEED=%d", seed)) seed = 101;
        if (!$value$plusargs("PATTERN=%d", pattern)) pattern = seed % 6;
        void'($urandom(seed));
        rst_n = 1'b0;
        load_valid = 1'b0;
        load_kind = '0;
        load_token = '0;
        load_dim = '0;
        load_data = '0;
        start_valid = 1'b0;
        start_seq_len = SEQ_LEN_W'(`HW_H9_SEQ_LEN);
        start_meta = 16'hB000;
        source_gap_cycles = 0;
        total_outputs = 0;
        total_done = 0;
        apply_reset();

        for (int tok = 0; tok < `HW_H9_TOKEN_COUNT; tok++) begin
            run_one(tok);
            repeat ($urandom_range(1, 5)) @(posedge clk);
        end

        if (total_done != `HW_H9_TOKEN_COUNT) fail("done count mismatch");
        if (total_outputs != (`HW_H9_TOKEN_COUNT * EXPECTED_OUTPUTS)) fail("total output count mismatch");
        if (perf_qk_sfu_overlap_cycles == 0) fail("missing QK-SFU overlap");
        if (perf_sfu_sv_overlap_cycles == 0) fail("missing SFU-sV overlap");
        $display("HW_H9_RANDOM_BACKPRESSURE_PASS seed=%0d pattern=%0d config=H1/D%0d seq=%0d token_count=%0d cycles=%0d watchdog=%0d source_gap=%0d output_stall=%0d done_stall=%0d score_peak=%0d prob_peak=%0d score_full_stall=%0d score_empty=%0d prob_full_stall=%0d prob_empty=%0d qk_sfu_overlap=%0d sfu_sv_overlap=%0d outputs=%0d done=%0d",
                 seed, pattern, D_HEAD, `HW_H9_SEQ_LEN, `HW_H9_TOKEN_COUNT,
                 cycle_counter, WATCHDOG_LIMIT, source_gap_cycles,
                 output_stall_cycles, done_stall_cycles,
                 perf_score_fifo_peak_occupancy, perf_probability_fifo_peak_occupancy,
                 perf_score_fifo_full_stall_cycles, perf_score_fifo_empty_cycles,
                 perf_probability_fifo_full_stall_cycles,
                 perf_probability_fifo_empty_stall_cycles,
                 perf_qk_sfu_overlap_cycles, perf_sfu_sv_overlap_cycles,
                 total_outputs, total_done);
        $finish;
    end
endmodule

`default_nettype wire
