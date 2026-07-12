`timescale 1ns/1ps
`default_nettype none

module tb_shared_gemv_projection_core;
    localparam int D_MODEL = 32;
    localparam int PE_NUM = 8;
    localparam int META_W = 16;
    localparam int COUNTER_W = 64;
    localparam int MAX_CASES = 128;
    localparam int DIM_W = $clog2(D_MODEL);
    localparam int LEN_W = $clog2(D_MODEL + 1);

    logic clk;
    logic rst_n;
    logic command_valid;
    logic command_ready;
    logic [1:0] command_matrix_kind;
    logic [LEN_W-1:0] command_input_length;
    logic [DIM_W-1:0] command_output_index;
    logic [D_MODEL*16-1:0] command_input_vector_fp16;
    logic [D_MODEL*16-1:0] command_weight_row_fp16;
    logic [META_W-1:0] command_meta;
    logic command_last;
    logic output_valid;
    logic output_ready;
    logic [1:0] output_matrix_kind;
    logic [DIM_W-1:0] output_index;
    logic [31:0] output_data_fp32;
    logic [PE_NUM-1:0] output_lane_mask;
    logic [7:0] output_status;
    logic output_invalid;
    logic [META_W-1:0] output_meta;
    logic output_last;
    logic [COUNTER_W-1:0] perf_total_cycles;
    logic [COUNTER_W-1:0] perf_tile_cycles;
    logic [COUNTER_W-1:0] perf_pe_stall_cycles;
    logic [COUNTER_W-1:0] perf_output_stall_cycles;

    int vec_len [0:MAX_CASES-1];
    int vec_out_idx [0:MAX_CASES-1];
    logic [META_W-1:0] vec_meta [0:MAX_CASES-1];
    logic [31:0] vec_expected [0:MAX_CASES-1];
    logic vec_invalid [0:MAX_CASES-1];
    logic vec_last [0:MAX_CASES-1];
    logic [15:0] vec_input [0:MAX_CASES-1][0:D_MODEL-1];
    logic [15:0] vec_weight [0:MAX_CASES-1][0:D_MODEL-1];
    int vector_count;

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    shared_gemv_projection_core #(
        .D_MODEL(D_MODEL),
        .PE_NUM(PE_NUM),
        .META_W(META_W),
        .COUNTER_W(COUNTER_W),
        .ASSERT_ON_INVALID(1'b0)
    ) u_dut (
        .clk                       (clk),
        .rst_n                     (rst_n),
        .command_valid             (command_valid),
        .command_ready             (command_ready),
        .command_matrix_kind       (command_matrix_kind),
        .command_input_length      (command_input_length),
        .command_output_index      (command_output_index),
        .command_input_vector_fp16 (command_input_vector_fp16),
        .command_weight_row_fp16   (command_weight_row_fp16),
        .command_meta              (command_meta),
        .command_last              (command_last),
        .output_valid              (output_valid),
        .output_ready              (output_ready),
        .output_matrix_kind        (output_matrix_kind),
        .output_index              (output_index),
        .output_data_fp32          (output_data_fp32),
        .output_lane_mask          (output_lane_mask),
        .output_status             (output_status),
        .output_invalid            (output_invalid),
        .output_meta               (output_meta),
        .output_last               (output_last),
        .perf_total_cycles         (perf_total_cycles),
        .perf_tile_cycles          (perf_tile_cycles),
        .perf_pe_stall_cycles      (perf_pe_stall_cycles),
        .perf_output_stall_cycles  (perf_output_stall_cycles)
    );

    task automatic tb_fail(input string message);
        begin
            $display("STAGE6B_GEMV_FAIL: %s", message);
            $fatal(1);
        end
    endtask

    task automatic load_vectors;
        string path;
        int fd;
        int code;
        int length;
        int out_idx;
        logic [META_W-1:0] meta;
        logic [31:0] expected;
        int invalid;
        int last;
        logic [15:0] input_values [0:D_MODEL-1];
        logic [15:0] weight_values [0:D_MODEL-1];
        begin
            if (!$value$plusargs("GEMV_VECTOR_FILE=%s", path)) tb_fail("missing GEMV_VECTOR_FILE");
            fd = $fopen(path, "r");
            if (fd == 0) tb_fail("could not open GEMV vector file");
            vector_count = 0;
            while (!$feof(fd)) begin
                code = $fscanf(fd,
                    "%h %h %h %h %d %d %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h\n",
                    length, out_idx, meta, expected, invalid, last,
                    input_values[0], input_values[1], input_values[2], input_values[3],
                    input_values[4], input_values[5], input_values[6], input_values[7],
                    input_values[8], input_values[9], input_values[10], input_values[11],
                    input_values[12], input_values[13], input_values[14], input_values[15],
                    input_values[16], input_values[17], input_values[18], input_values[19],
                    input_values[20], input_values[21], input_values[22], input_values[23],
                    input_values[24], input_values[25], input_values[26], input_values[27],
                    input_values[28], input_values[29], input_values[30], input_values[31],
                    weight_values[0], weight_values[1], weight_values[2], weight_values[3],
                    weight_values[4], weight_values[5], weight_values[6], weight_values[7],
                    weight_values[8], weight_values[9], weight_values[10], weight_values[11],
                    weight_values[12], weight_values[13], weight_values[14], weight_values[15],
                    weight_values[16], weight_values[17], weight_values[18], weight_values[19],
                    weight_values[20], weight_values[21], weight_values[22], weight_values[23],
                    weight_values[24], weight_values[25], weight_values[26], weight_values[27],
                    weight_values[28], weight_values[29], weight_values[30], weight_values[31]);
                if (code == 70) begin
                    if (vector_count >= MAX_CASES) tb_fail("too many GEMV vectors");
                    vec_len[vector_count] = length;
                    vec_out_idx[vector_count] = out_idx;
                    vec_meta[vector_count] = meta;
                    vec_expected[vector_count] = expected;
                    vec_invalid[vector_count] = invalid != 0;
                    vec_last[vector_count] = last != 0;
                    for (int idx = 0; idx < D_MODEL; idx++) begin
                        vec_input[vector_count][idx] = input_values[idx];
                        vec_weight[vector_count][idx] = weight_values[idx];
                    end
                    vector_count++;
                end
            end
            $fclose(fd);
            if (vector_count == 0) tb_fail("no GEMV vectors loaded");
        end
    endtask

    task automatic apply_reset;
        begin
            rst_n = 1'b0;
            command_valid = 1'b0;
            command_matrix_kind = 2'd0;
            command_input_length = '0;
            command_output_index = '0;
            command_input_vector_fp16 = '0;
            command_weight_row_fp16 = '0;
            command_meta = '0;
            command_last = 1'b0;
            output_ready = 1'b0;
            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
            if (output_valid) tb_fail("output valid not clear after reset");
        end
    endtask

    task automatic pack_case(input int case_idx);
        begin
            command_input_vector_fp16 = '0;
            command_weight_row_fp16 = '0;
            for (int idx = 0; idx < D_MODEL; idx++) begin
                command_input_vector_fp16[idx*16 +: 16] = vec_input[case_idx][idx];
                command_weight_row_fp16[idx*16 +: 16] = vec_weight[case_idx][idx];
            end
        end
    endtask

    task automatic send_case(input int case_idx);
        int cycle;
        logic sent;
        logic received;
        logic drive_valid;
        logic pre_command_fire;
        logic pre_output_fire;
        logic [1:0] pre_output_matrix_kind;
        logic [DIM_W-1:0] pre_output_index;
        logic [31:0] pre_output_data_fp32;
        logic [7:0] pre_output_status;
        logic pre_output_invalid;
        logic [META_W-1:0] pre_output_meta;
        logic pre_output_last;
        begin
            cycle = 0;
            sent = 1'b0;
            received = 1'b0;
            drive_valid = 1'b0;
            pack_case(case_idx);
            while (!received) begin
                @(negedge clk);
                if (!drive_valid && !sent && ((cycle % 4) != 1)) begin
                    drive_valid = 1'b1;
                end
                command_valid = drive_valid;
                command_matrix_kind = 2'd0;
                command_input_length = LEN_W'(vec_len[case_idx]);
                command_output_index = DIM_W'(vec_out_idx[case_idx]);
                command_meta = vec_meta[case_idx];
                command_last = vec_last[case_idx];
                output_ready = ((cycle % 5) != 2) && ((cycle % 7) != 3);
                #1;
                pre_command_fire = command_valid && command_ready;
                pre_output_fire = output_valid && output_ready;
                pre_output_matrix_kind = output_matrix_kind;
                pre_output_index = output_index;
                pre_output_data_fp32 = output_data_fp32;
                pre_output_status = output_status;
                pre_output_invalid = output_invalid;
                pre_output_meta = output_meta;
                pre_output_last = output_last;
                @(posedge clk); #1;
                if (pre_command_fire) begin
                    sent = 1'b1;
                    drive_valid = 1'b0;
                    command_valid = 1'b0;
                end
                if (pre_output_fire) begin
                    if (!sent) tb_fail("output before command accepted");
                    if (pre_output_data_fp32 !== vec_expected[case_idx]) begin
                        $display("CHECK_FAIL gemv data case=%0d got=%08h expected=%08h",
                                 case_idx, pre_output_data_fp32, vec_expected[case_idx]);
                        $fatal(1);
                    end
                    if (pre_output_index !== DIM_W'(vec_out_idx[case_idx])) tb_fail("output index mismatch");
                    if (pre_output_meta !== vec_meta[case_idx]) tb_fail("metadata mismatch");
                    if (pre_output_last !== vec_last[case_idx]) tb_fail("last mismatch");
                    if (pre_output_invalid !== vec_invalid[case_idx]) tb_fail("invalid mismatch");
                    received = 1'b1;
                end
                cycle++;
                if (cycle > 2000) tb_fail("GEMV case timeout");
            end
        end
    endtask

    initial begin
        load_vectors();
        apply_reset();
        for (int idx = 0; idx < vector_count; idx++) begin
            send_case(idx);
        end
        $display("STAGE6B_SHARED_GEMV_PASS cases=%0d total_cycles=%0d tile_cycles=%0d pe_stall=%0d output_stall=%0d",
                 vector_count, perf_total_cycles, perf_tile_cycles, perf_pe_stall_cycles, perf_output_stall_cycles);
        $finish;
    end
endmodule

`default_nettype wire
