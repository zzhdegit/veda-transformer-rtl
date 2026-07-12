`timescale 1ns/1ps
`default_nettype none

module tb_reconfigurable_pe_core;
    localparam int PE_NUM = 8;
    localparam int META_W = 16;
    localparam int MAX_VECTORS = 64;

    logic clk;
    logic rst_n;
    logic in_valid;
    logic in_ready;
    logic [1:0] in_mode;
    logic in_clear;
    logic in_tile_first;
    logic in_tile_last;
    logic in_use_explicit_mask;
    logic [$clog2(PE_NUM+1)-1:0] in_active_lanes;
    logic [PE_NUM-1:0] in_lane_mask;
    logic [31:0] in_scalar_fp32;
    logic [PE_NUM*16-1:0] in_vector_a_fp16;
    logic [PE_NUM*16-1:0] in_vector_b_fp16;
    logic [META_W-1:0] in_meta;
    logic in_last;

    logic out_valid;
    logic out_ready;
    logic [1:0] out_mode;
    logic [31:0] out_scalar_fp32;
    logic [PE_NUM*32-1:0] out_vector_fp32;
    logic [PE_NUM-1:0] out_lane_mask;
    logic [7:0] out_status;
    logic out_invalid;
    logic [META_W-1:0] out_meta;
    logic out_last;

    logic [63:0] perf_total_cycles;
    logic [63:0] perf_busy_cycles;
    logic [63:0] perf_active_lane_cycles;
    logic [63:0] perf_available_lane_cycles;
    logic [63:0] perf_input_stall_cycles;
    logic [63:0] perf_output_stall_cycles;
    logic [63:0] perf_mode_switch_cycles;
    logic [63:0] perf_tile_count;
    logic [63:0] perf_operation_count;
    logic [63:0] perf_invalid_count;

    logic [1:0] vec_mode [0:MAX_VECTORS-1];
    logic vec_clear [0:MAX_VECTORS-1];
    logic vec_first [0:MAX_VECTORS-1];
    logic vec_last_tile [0:MAX_VECTORS-1];
    logic [PE_NUM-1:0] vec_mask [0:MAX_VECTORS-1];
    logic [31:0] vec_scalar [0:MAX_VECTORS-1];
    logic [PE_NUM*16-1:0] vec_a [0:MAX_VECTORS-1];
    logic [PE_NUM*16-1:0] vec_b [0:MAX_VECTORS-1];
    logic vec_expect [0:MAX_VECTORS-1];
    logic [31:0] vec_exp_scalar [0:MAX_VECTORS-1];
    logic [PE_NUM*32-1:0] vec_exp_vector [0:MAX_VECTORS-1];
    logic [META_W-1:0] vec_meta [0:MAX_VECTORS-1];
    logic vec_last [0:MAX_VECTORS-1];
    int vector_count;

    logic [1:0] exp_mode_q[$];
    logic [31:0] exp_scalar_q[$];
    logic [PE_NUM*32-1:0] exp_vector_q[$];
    logic [PE_NUM-1:0] exp_mask_q[$];
    logic [META_W-1:0] exp_meta_q[$];
    logic exp_last_q[$];

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    reconfigurable_pe_core #(
        .PE_NUM(PE_NUM),
        .META_W(META_W)
    ) u_dut (
        .clk                         (clk),
        .rst_n                       (rst_n),
        .in_valid                    (in_valid),
        .in_ready                    (in_ready),
        .in_mode                     (in_mode),
        .in_clear                    (in_clear),
        .in_tile_first               (in_tile_first),
        .in_tile_last                (in_tile_last),
        .in_use_explicit_mask        (in_use_explicit_mask),
        .in_active_lanes             (in_active_lanes),
        .in_lane_mask                (in_lane_mask),
        .in_scalar_fp32              (in_scalar_fp32),
        .in_vector_a_fp16            (in_vector_a_fp16),
        .in_vector_b_fp16            (in_vector_b_fp16),
        .in_meta                     (in_meta),
        .in_last                     (in_last),
        .out_valid                   (out_valid),
        .out_ready                   (out_ready),
        .out_mode                    (out_mode),
        .out_scalar_fp32             (out_scalar_fp32),
        .out_vector_fp32             (out_vector_fp32),
        .out_lane_mask               (out_lane_mask),
        .out_status                  (out_status),
        .out_invalid                 (out_invalid),
        .out_meta                    (out_meta),
        .out_last                    (out_last),
        .perf_total_cycles           (perf_total_cycles),
        .perf_busy_cycles            (perf_busy_cycles),
        .perf_active_lane_cycles     (perf_active_lane_cycles),
        .perf_available_lane_cycles  (perf_available_lane_cycles),
        .perf_input_stall_cycles     (perf_input_stall_cycles),
        .perf_output_stall_cycles    (perf_output_stall_cycles),
        .perf_mode_switch_cycles     (perf_mode_switch_cycles),
        .perf_tile_count             (perf_tile_count),
        .perf_operation_count        (perf_operation_count),
        .perf_invalid_count          (perf_invalid_count)
    );

    task automatic tb_fail(input string message);
        begin
            $display("STAGE2_PE_CORE_TB_FAIL: %s", message);
            $fatal(1);
        end
    endtask

    task automatic load_vectors;
        string path;
        int fd;
        int code;
        logic [1:0] mode;
        logic clear;
        logic first;
        logic last_tile;
        logic [PE_NUM-1:0] mask;
        logic [31:0] scalar;
        logic [15:0] a [0:PE_NUM-1];
        logic [15:0] b [0:PE_NUM-1];
        logic expect_out;
        logic [31:0] exp_scalar;
        logic [31:0] exp_vector [0:PE_NUM-1];
        logic [META_W-1:0] meta;
        logic last;
        begin
            if (!$value$plusargs("CORE_VECTOR_FILE=%s", path)) tb_fail("missing +CORE_VECTOR_FILE");
            fd = $fopen(path, "r");
            if (fd == 0) tb_fail("could not open core vector file");
            vector_count = 0;
            while (!$feof(fd)) begin
                code = $fscanf(fd,
                    "%h %b %b %b %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %b %h %h %h %h %h %h %h %h %h %h %b\n",
                    mode, clear, first, last_tile, mask, scalar,
                    a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
                    b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                    expect_out, exp_scalar,
                    exp_vector[0], exp_vector[1], exp_vector[2], exp_vector[3],
                    exp_vector[4], exp_vector[5], exp_vector[6], exp_vector[7],
                    meta, last);
                if (code == 34) begin
                    vec_mode[vector_count] = mode;
                    vec_clear[vector_count] = clear;
                    vec_first[vector_count] = first;
                    vec_last_tile[vector_count] = last_tile;
                    vec_mask[vector_count] = mask;
                    vec_scalar[vector_count] = scalar;
                    for (int lane = 0; lane < PE_NUM; lane++) begin
                        vec_a[vector_count][lane*16 +: 16] = a[lane];
                        vec_b[vector_count][lane*16 +: 16] = b[lane];
                        vec_exp_vector[vector_count][lane*32 +: 32] = exp_vector[lane];
                    end
                    vec_expect[vector_count] = expect_out;
                    vec_exp_scalar[vector_count] = exp_scalar;
                    vec_meta[vector_count] = meta;
                    vec_last[vector_count] = last;
                    vector_count++;
                end
            end
            $fclose(fd);
            if (vector_count == 0) tb_fail("no core vectors loaded");
            $display("STAGE2_PE_CORE_VECTORS count=%0d", vector_count);
        end
    endtask

    task automatic apply_reset;
        begin
            rst_n = 1'b0;
            in_valid = 1'b0;
            in_mode = 2'd0;
            in_clear = 1'b0;
            in_tile_first = 1'b0;
            in_tile_last = 1'b0;
            in_use_explicit_mask = 1'b1;
            in_active_lanes = '0;
            in_lane_mask = '0;
            in_scalar_fp32 = '0;
            in_vector_a_fp16 = '0;
            in_vector_b_fp16 = '0;
            in_meta = '0;
            in_last = 1'b0;
            out_ready = 1'b0;
            repeat (6) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
            if (out_valid) tb_fail("out_valid not clear after reset");
        end
    endtask

    task automatic run_vectors;
        int sent;
        int received;
        int expected_outputs;
        int cycle;
        int drive_index;
        logic drive_valid;
        logic pre_in_fire;
        logic pre_out_fire;
        logic [1:0] pre_mode;
        logic [31:0] pre_scalar;
        logic [PE_NUM*32-1:0] pre_vector;
        logic [PE_NUM-1:0] pre_mask;
        logic [7:0] pre_status;
        logic pre_invalid;
        logic [META_W-1:0] pre_meta;
        logic pre_last;
        logic [1:0] exp_mode;
        logic [31:0] exp_scalar;
        logic [PE_NUM*32-1:0] exp_vector;
        begin
            sent = 0;
            received = 0;
            expected_outputs = 0;
            for (int idx = 0; idx < vector_count; idx++) begin
                expected_outputs += vec_expect[idx];
            end
            cycle = 0;
            drive_valid = 1'b0;
            drive_index = 0;
            while (received < expected_outputs) begin
                @(negedge clk);
                if (!drive_valid && (sent < vector_count) && ((cycle % 4) != 2)) begin
                    drive_valid = 1'b1;
                    drive_index = sent;
                end
                in_valid = drive_valid;
                in_mode = vec_mode[drive_index];
                in_clear = vec_clear[drive_index];
                in_tile_first = vec_first[drive_index];
                in_tile_last = vec_last_tile[drive_index];
                in_use_explicit_mask = 1'b1;
                in_active_lanes = '0;
                in_lane_mask = vec_mask[drive_index];
                in_scalar_fp32 = vec_scalar[drive_index];
                in_vector_a_fp16 = vec_a[drive_index];
                in_vector_b_fp16 = vec_b[drive_index];
                in_meta = vec_meta[drive_index];
                in_last = vec_last[drive_index];
                out_ready = ((cycle % 6) != 3) && ((cycle % 11) != 7);
                #1;
                pre_in_fire = in_valid && in_ready;
                pre_out_fire = out_valid && out_ready;
                pre_mode = out_mode;
                pre_scalar = out_scalar_fp32;
                pre_vector = out_vector_fp32;
                pre_mask = out_lane_mask;
                pre_status = out_status;
                pre_invalid = out_invalid;
                pre_meta = out_meta;
                pre_last = out_last;
                @(posedge clk); #1;
                if (pre_out_fire) begin
                    if (exp_mode_q.size() == 0) tb_fail("unexpected core output");
                    exp_mode = exp_mode_q.pop_front();
                    exp_scalar = exp_scalar_q.pop_front();
                    exp_vector = exp_vector_q.pop_front();
                    if (pre_mode !== exp_mode) tb_fail("mode mismatch");
                    if (exp_mode == 2'd2) begin
                        if (pre_vector !== exp_vector) begin
                            $display("CHECK_FAIL core outer vector got=%h expected=%h", pre_vector, exp_vector);
                            $fatal(1);
                        end
                    end else begin
                        if (pre_scalar !== exp_scalar) begin
                            $display("CHECK_FAIL core scalar got=%08h expected=%08h", pre_scalar, exp_scalar);
                            $fatal(1);
                        end
                    end
                    if (pre_mask !== exp_mask_q.pop_front()) tb_fail("lane mask mismatch");
                    if (pre_invalid) tb_fail("unexpected core invalid");
                    if (^pre_status === 1'bx) tb_fail("unknown core status");
                    if (pre_meta !== exp_meta_q.pop_front()) tb_fail("metadata mismatch");
                    if (pre_last !== exp_last_q.pop_front()) tb_fail("last mismatch");
                    received++;
                end
                if (pre_in_fire) begin
                    if (vec_expect[drive_index]) begin
                        exp_mode_q.push_back(vec_mode[drive_index]);
                        exp_scalar_q.push_back(vec_exp_scalar[drive_index]);
                        exp_vector_q.push_back(vec_exp_vector[drive_index]);
                        exp_mask_q.push_back(vec_mask[drive_index]);
                        exp_meta_q.push_back(vec_meta[drive_index]);
                        exp_last_q.push_back(vec_last[drive_index]);
                    end
                    sent++;
                    drive_valid = 1'b0;
                end
                cycle++;
                if (cycle > 200000) tb_fail("core timeout");
            end
            if (sent != vector_count) tb_fail("not all core vectors sent");
            if (perf_tile_count < 64'(vector_count)) tb_fail("perf tile count too small");
            if (perf_operation_count < 64'(expected_outputs)) tb_fail("perf operation count too small");
            if (perf_available_lane_cycles == 64'd0) tb_fail("perf available lane slots not counted");
            $display("STAGE2_PE_CORE_PERF total=%0d busy=%0d active_lane_cycles=%0d available_lane_cycles=%0d input_stall=%0d output_stall=%0d mode_switch=%0d tile_count=%0d operation_count=%0d invalid_count=%0d",
                     perf_total_cycles, perf_busy_cycles, perf_active_lane_cycles, perf_available_lane_cycles,
                     perf_input_stall_cycles, perf_output_stall_cycles, perf_mode_switch_cycles,
                     perf_tile_count, perf_operation_count, perf_invalid_count);
        end
    endtask

    initial begin
        load_vectors();
        apply_reset();
        run_vectors();
        $display("STAGE2_RECONFIGURABLE_PE_CORE_PASS");
        $finish;
    end
endmodule

`default_nettype wire
