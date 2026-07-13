`timescale 1ns/1ps
`default_nettype none

`ifndef HW_H9_D_HEAD
`define HW_H9_D_HEAD 8
`endif

`ifndef HW_H9_SEQ_LEN
`define HW_H9_SEQ_LEN 8
`endif

`ifndef HW_H9_SCHEDULE
`define HW_H9_SCHEDULE 0
`endif

module tb_h9_matched_ab_single_head;
    localparam int PE_NUM = 8;
    localparam int D_HEAD = `HW_H9_D_HEAD;
    localparam int MAX_SEQ_LEN = 64;
    localparam int META_W = 16;
    localparam int COUNTER_W = 64;
    localparam int ATTENTION_PE_ARCH = 1;
    localparam int ATTENTION_SCHEDULE = `HW_H9_SCHEDULE;
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

    logic [COUNTER_W-1:0] h9_qk_sfu_overlap_cycles;
    logic [COUNTER_W-1:0] h9_sfu_sv_overlap_cycles;
    logic [COUNTER_W-1:0] h9_score_fifo_peak_occupancy;
    logic [COUNTER_W-1:0] h9_probability_fifo_peak_occupancy;
    logic [COUNTER_W-1:0] h9_pipeline_bubble_cycles;
    logic [COUNTER_W-1:0] h9_score_fifo_full_stall_cycles;
    logic [COUNTER_W-1:0] h9_score_fifo_empty_cycles;
    logic [COUNTER_W-1:0] h9_probability_fifo_full_stall_cycles;
    logic [COUNTER_W-1:0] h9_probability_fifo_empty_stall_cycles;

    longint unsigned cycle_counter;
    longint unsigned start_cycle;
    longint unsigned output_cycle;
    longint unsigned done_cycle;

    single_head_attention #(
        .PE_NUM(PE_NUM),
        .D_HEAD(D_HEAD),
        .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .META_W(META_W),
        .COUNTER_W(COUNTER_W),
        .ATTENTION_PE_ARCH(ATTENTION_PE_ARCH),
        .ATTENTION_SCHEDULE(ATTENTION_SCHEDULE),
        .ASSERT_ON_INVALID(1'b1)
    ) dut (
        .clk                              (clk),
        .rst_n                            (rst_n),
        .load_valid                       (load_valid),
        .load_ready                       (load_ready),
        .load_kind                        (load_kind),
        .load_token                       (load_token),
        .load_dim                         (load_dim),
        .load_data                        (load_data),
        .start_valid                      (start_valid),
        .start_ready                      (start_ready),
        .start_seq_len                    (start_seq_len),
        .start_meta                       (start_meta),
        .output_valid                     (output_valid),
        .output_ready                     (output_ready),
        .output_base_dim                  (output_base_dim),
        .output_vector_fp32               (output_vector_fp32),
        .output_lane_mask                 (output_lane_mask),
        .output_status                    (output_status),
        .output_invalid                   (output_invalid),
        .output_meta                      (output_meta),
        .output_last                      (output_last),
        .done_valid                       (done_valid),
        .done_ready                       (done_ready),
        .done_status                      (done_status),
        .done_invalid                     (done_invalid),
        .done_meta                        (done_meta),
        .perf_total_attention_cycles      (perf_total_attention_cycles),
        .perf_qk_cycles                   (perf_qk_cycles),
        .perf_qk_pe_busy_cycles           (perf_qk_pe_busy_cycles),
        .perf_scale_cycles                (perf_scale_cycles),
        .perf_reduction_cycles            (perf_reduction_cycles),
        .perf_reduction_finalize_cycles   (perf_reduction_finalize_cycles),
        .perf_normalization_cycles        (perf_normalization_cycles),
        .perf_sv_cycles                   (perf_sv_cycles),
        .perf_pe_stall_cycles             (perf_pe_stall_cycles),
        .perf_sfu_stall_cycles            (perf_sfu_stall_cycles),
        .perf_buffer_stall_cycles         (perf_buffer_stall_cycles),
        .perf_output_stall_cycles         (perf_output_stall_cycles),
        .perf_score_buffer_peak_occupancy (perf_score_buffer_peak_occupancy),
        .perf_paper_array_active_cycles   (perf_paper_array_active_cycles),
        .perf_paper_array_idle_cycles     (perf_paper_array_idle_cycles),
        .perf_inner_mode_cycles           (perf_inner_mode_cycles),
        .perf_outer_mode_cycles           (perf_outer_mode_cycles),
        .perf_group0_active_cycles        (perf_group0_active_cycles),
        .perf_group1_active_cycles        (perf_group1_active_cycles),
        .perf_tail_masked_pe_cycles       (perf_tail_masked_pe_cycles),
        .perf_mode_switch_cycles          (perf_mode_switch_cycles),
        .perf_array_input_stall_cycles    (perf_array_input_stall_cycles),
        .perf_array_output_stall_cycles   (perf_array_output_stall_cycles)
    );

`ifdef HW_H9_INTERLEAVED
    assign h9_qk_sfu_overlap_cycles = dut.unused_h9_qk_sfu_overlap_cycles;
    assign h9_sfu_sv_overlap_cycles = dut.unused_h9_sfu_sv_overlap_cycles;
    assign h9_score_fifo_peak_occupancy = dut.unused_h9_score_fifo_peak_occupancy;
    assign h9_probability_fifo_peak_occupancy = dut.unused_h9_probability_fifo_peak_occupancy;
    assign h9_pipeline_bubble_cycles = dut.unused_h9_pipeline_bubble_cycles;
    assign h9_score_fifo_full_stall_cycles = dut.unused_h9_score_fifo_full_stall_cycles;
    assign h9_score_fifo_empty_cycles = dut.unused_h9_score_fifo_empty_cycles;
    assign h9_probability_fifo_full_stall_cycles = dut.unused_h9_probability_fifo_full_stall_cycles;
    assign h9_probability_fifo_empty_stall_cycles = dut.unused_h9_probability_fifo_empty_stall_cycles;
`else
    assign h9_qk_sfu_overlap_cycles = '0;
    assign h9_sfu_sv_overlap_cycles = '0;
    assign h9_score_fifo_peak_occupancy = '0;
    assign h9_probability_fifo_peak_occupancy = '0;
    assign h9_pipeline_bubble_cycles = '0;
    assign h9_score_fifo_full_stall_cycles = '0;
    assign h9_score_fifo_empty_cycles = '0;
    assign h9_probability_fifo_full_stall_cycles = '0;
    assign h9_probability_fifo_empty_stall_cycles = '0;
`endif

    initial clk = 1'b0;
    always #5 clk = ~clk;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_counter <= 0;
        end else begin
            cycle_counter <= cycle_counter + 1;
        end
    end

    always_comb begin
`ifdef HW_H9_DETERMINISTIC_BP
        output_ready = ((cycle_counter % 7) != 0) && ((cycle_counter % 11) != 0);
        done_ready = ((cycle_counter % 5) != 0);
`else
        output_ready = 1'b1;
        done_ready = 1'b1;
`endif
    end

    task automatic fail(input string msg);
        begin
            $display("HW_H9_MATCHED_AB_FAIL schedule=%0d d_head=%0d seq=%0d: %s",
                     ATTENTION_SCHEDULE, D_HEAD, `HW_H9_SEQ_LEN, msg);
            $finish;
        end
    endtask

    task automatic load_one(input logic [1:0] kind, input int token, input int dim, input logic [15:0] data);
        begin
            load_kind = kind;
            load_token = token[TOKEN_W-1:0];
            load_dim = dim[D_ADDR_W-1:0];
            load_data = data;
            do @(negedge clk); while (!load_ready);
            load_valid = 1'b1;
            @(negedge clk);
            load_valid = 1'b0;
        end
    endtask

    integer output_count;
    logic saw_last_output;

    initial begin
        rst_n = 1'b0;
        load_valid = 1'b0;
        load_kind = '0;
        load_token = '0;
        load_dim = '0;
        load_data = '0;
        start_valid = 1'b0;
        start_seq_len = SEQ_LEN_W'(`HW_H9_SEQ_LEN);
        start_meta = 16'h09AB;
        start_cycle = 0;
        output_cycle = 0;
        done_cycle = 0;
        output_count = 0;
        saw_last_output = 1'b0;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        if (`HW_H9_SEQ_LEN < 1 || `HW_H9_SEQ_LEN > MAX_SEQ_LEN) begin
            fail("illegal sequence length");
        end

        for (int dim = 0; dim < D_HEAD; dim++) begin
            load_one(2'd0, 0, dim, 16'h0000);
        end
        for (int token = 0; token < `HW_H9_SEQ_LEN; token++) begin
            for (int dim = 0; dim < D_HEAD; dim++) begin
                load_one(2'd1, token, dim, 16'h0000);
                load_one(2'd2, token, dim, 16'h3C00);
            end
        end

        do @(negedge clk); while (!start_ready);
        start_valid = 1'b1;
        start_cycle = cycle_counter;
        @(negedge clk);
        start_valid = 1'b0;

        fork
            begin : timeout_block
                repeat (1000000) @(posedge clk);
                fail("timeout");
            end
            begin : monitor_block
                forever begin
                    @(posedge clk);
                    if (output_valid && output_ready) begin
                        if (output_cycle == 0) begin
                            output_cycle = cycle_counter;
                        end
                        output_count++;
                        if (output_invalid) begin
                            fail("output invalid asserted");
                        end
                        if (output_meta !== start_meta) begin
                            fail("output metadata mismatch");
                        end
                        if (output_last) begin
                            saw_last_output = 1'b1;
                        end
                        for (int lane = 0; lane < PE_NUM; lane++) begin
                            if (output_lane_mask[lane] &&
                                output_vector_fp32[lane*32 +: 32] !== 32'h3F80_0000) begin
                                fail("output lane not FP32 one");
                            end
                        end
                    end
                    if (done_valid && done_ready) begin
                        done_cycle = cycle_counter;
                        if (done_invalid) fail("done invalid asserted");
                        if (done_meta !== start_meta) fail("done metadata mismatch");
                        if (output_count != EXPECTED_OUTPUTS) fail("unexpected output count");
                        if (!saw_last_output) fail("last output not marked");
                        $display("HW_H9_MATCHED_AB_PASS schedule=%0d d_head=%0d seq=%0d bp=%0d outputs=%0d start_cycle=%0d output_cycle=%0d done_cycle=%0d total=%0d qk=%0d qk_busy=%0d scale=%0d reduce=%0d reduce_final=%0d norm=%0d sv=%0d pe_stall=%0d sfu_stall=%0d buffer_stall=%0d output_stall=%0d score_peak=%0d array_active=%0d array_idle=%0d inner=%0d outer=%0d group0=%0d group1=%0d tail=%0d mode_switch=%0d array_in_stall=%0d array_out_stall=%0d qk_sfu_overlap=%0d sfu_sv_overlap=%0d h9_score_peak=%0d h9_prob_peak=%0d h9_pipeline_bubble=%0d h9_score_full_stall=%0d h9_score_empty=%0d h9_prob_full_stall=%0d h9_prob_empty=%0d",
                                 ATTENTION_SCHEDULE, D_HEAD, `HW_H9_SEQ_LEN,
`ifdef HW_H9_DETERMINISTIC_BP
                                 1,
`else
                                 0,
`endif
                                 output_count, start_cycle, output_cycle, done_cycle,
                                 perf_total_attention_cycles,
                                 perf_qk_cycles,
                                 perf_qk_pe_busy_cycles,
                                 perf_scale_cycles,
                                 perf_reduction_cycles,
                                 perf_reduction_finalize_cycles,
                                 perf_normalization_cycles,
                                 perf_sv_cycles,
                                 perf_pe_stall_cycles,
                                 perf_sfu_stall_cycles,
                                 perf_buffer_stall_cycles,
                                 perf_output_stall_cycles,
                                 perf_score_buffer_peak_occupancy,
                                 perf_paper_array_active_cycles,
                                 perf_paper_array_idle_cycles,
                                 perf_inner_mode_cycles,
                                 perf_outer_mode_cycles,
                                 perf_group0_active_cycles,
                                 perf_group1_active_cycles,
                                 perf_tail_masked_pe_cycles,
                                 perf_mode_switch_cycles,
                                 perf_array_input_stall_cycles,
                                 perf_array_output_stall_cycles,
                                 h9_qk_sfu_overlap_cycles,
                                 h9_sfu_sv_overlap_cycles,
                                 h9_score_fifo_peak_occupancy,
                                 h9_probability_fifo_peak_occupancy,
                                 h9_pipeline_bubble_cycles,
                                 h9_score_fifo_full_stall_cycles,
                                 h9_score_fifo_empty_cycles,
                                 h9_probability_fifo_full_stall_cycles,
                                 h9_probability_fifo_empty_stall_cycles);
                        $finish;
                    end
                end
            end
        join
    end
endmodule

`default_nettype wire
