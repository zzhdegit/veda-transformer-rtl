`timescale 1ns/1ps
`default_nettype none

module tb_pe_lane;
    localparam int META_W = 16;
    localparam int MAX_VECTORS = 64;

    logic clk;
    logic rst_n;
    logic in_valid;
    logic in_ready;
    logic in_mode;
    logic in_lane_enable;
    logic in_lane_mask;
    logic [31:0] in_scalar;
    logic [31:0] in_vector;
    logic [31:0] in_accumulator;
    logic [META_W-1:0] in_meta;
    logic in_last;
    logic out_valid;
    logic out_ready;
    logic [31:0] out_result;
    logic [7:0] out_status;
    logic out_invalid;
    logic out_lane_active;
    logic [META_W-1:0] out_meta;
    logic out_last;

    logic vec_mode [0:MAX_VECTORS-1];
    logic vec_active [0:MAX_VECTORS-1];
    logic [31:0] vec_scalar [0:MAX_VECTORS-1];
    logic [31:0] vec_vector [0:MAX_VECTORS-1];
    logic [31:0] vec_acc [0:MAX_VECTORS-1];
    logic [31:0] vec_expected [0:MAX_VECTORS-1];
    logic [META_W-1:0] vec_meta [0:MAX_VECTORS-1];
    logic vec_last [0:MAX_VECTORS-1];
    int vector_count;

    logic [31:0] exp_result_q[$];
    logic exp_active_q[$];
    logic [META_W-1:0] exp_meta_q[$];
    logic exp_last_q[$];

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    pe_lane #(
        .META_W(META_W)
    ) u_dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .in_valid       (in_valid),
        .in_ready       (in_ready),
        .in_mode        (in_mode),
        .in_lane_enable (in_lane_enable),
        .in_lane_mask   (in_lane_mask),
        .in_scalar      (in_scalar),
        .in_vector      (in_vector),
        .in_accumulator (in_accumulator),
        .in_meta        (in_meta),
        .in_last        (in_last),
        .out_valid      (out_valid),
        .out_ready      (out_ready),
        .out_result     (out_result),
        .out_status     (out_status),
        .out_invalid    (out_invalid),
        .out_lane_active(out_lane_active),
        .out_meta       (out_meta),
        .out_last       (out_last)
    );

    task automatic tb_fail(input string message);
        begin
            $display("STAGE2_PE_LANE_TB_FAIL: %s", message);
            $fatal(1);
        end
    endtask

    task automatic load_vectors;
        string path;
        int fd;
        int code;
        logic mode;
        logic active;
        logic [31:0] scalar;
        logic [31:0] vector;
        logic [31:0] acc;
        logic [31:0] expected;
        logic [META_W-1:0] meta;
        logic last;
        begin
            if (!$value$plusargs("LANE_VECTOR_FILE=%s", path)) tb_fail("missing +LANE_VECTOR_FILE");
            fd = $fopen(path, "r");
            if (fd == 0) tb_fail("could not open lane vector file");
            vector_count = 0;
            while (!$feof(fd)) begin
                code = $fscanf(fd, "%b %b %h %h %h %h %h %b\n",
                               mode, active, scalar, vector, acc, expected, meta, last);
                if (code == 8) begin
                    vec_mode[vector_count] = mode;
                    vec_active[vector_count] = active;
                    vec_scalar[vector_count] = scalar;
                    vec_vector[vector_count] = vector;
                    vec_acc[vector_count] = acc;
                    vec_expected[vector_count] = expected;
                    vec_meta[vector_count] = meta;
                    vec_last[vector_count] = last;
                    vector_count++;
                end
            end
            $fclose(fd);
            if (vector_count == 0) tb_fail("no lane vectors loaded");
            $display("STAGE2_PE_LANE_VECTORS count=%0d", vector_count);
        end
    endtask

    task automatic apply_reset;
        begin
            rst_n = 1'b0;
            in_valid = 1'b0;
            in_mode = 1'b0;
            in_lane_enable = 1'b1;
            in_lane_mask = 1'b0;
            in_scalar = '0;
            in_vector = '0;
            in_accumulator = '0;
            in_meta = '0;
            in_last = 1'b0;
            out_ready = 1'b0;
            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
            if (out_valid) tb_fail("out_valid not clear after reset");
        end
    endtask

    task automatic run_vectors;
        int sent;
        int received;
        int cycle;
        int drive_index;
        logic drive_valid;
        logic pre_in_fire;
        logic pre_out_fire;
        logic [31:0] pre_result;
        logic [7:0] pre_status;
        logic pre_invalid;
        logic pre_active;
        logic [META_W-1:0] pre_meta;
        logic pre_last;
        logic [31:0] expected;
        logic active;
        begin
            sent = 0;
            received = 0;
            cycle = 0;
            drive_valid = 1'b0;
            drive_index = 0;
            while (received < vector_count) begin
                @(negedge clk);
                if (!drive_valid && (sent < vector_count) && ((cycle % 3) != 1)) begin
                    drive_valid = 1'b1;
                    drive_index = sent;
                end
                in_valid = drive_valid;
                in_mode = vec_mode[drive_index];
                in_lane_enable = 1'b1;
                in_lane_mask = vec_active[drive_index];
                in_scalar = vec_scalar[drive_index];
                in_vector = vec_vector[drive_index];
                in_accumulator = vec_acc[drive_index];
                in_meta = vec_meta[drive_index];
                in_last = vec_last[drive_index];
                out_ready = ((cycle % 5) != 0);
                #1;
                pre_in_fire = in_valid && in_ready;
                pre_out_fire = out_valid && out_ready;
                pre_result = out_result;
                pre_status = out_status;
                pre_invalid = out_invalid;
                pre_active = out_lane_active;
                pre_meta = out_meta;
                pre_last = out_last;
                @(posedge clk); #1;
                if (pre_out_fire) begin
                    if (exp_result_q.size() == 0) tb_fail("unexpected lane output");
                    expected = exp_result_q.pop_front();
                    active = exp_active_q.pop_front();
                    if (pre_result !== expected) begin
                        $display("CHECK_FAIL pe_lane result got=%08h expected=%08h", pre_result, expected);
                        $fatal(1);
                    end
                    if (pre_invalid) tb_fail("unexpected lane invalid");
                    if (^pre_status === 1'bx) tb_fail("unknown lane status");
                    if (pre_active !== active) tb_fail("lane active mismatch");
                    if (pre_meta !== exp_meta_q.pop_front()) tb_fail("metadata mismatch");
                    if (pre_last !== exp_last_q.pop_front()) tb_fail("last mismatch");
                    received++;
                end
                if (pre_in_fire) begin
                    exp_result_q.push_back(vec_expected[drive_index]);
                    exp_active_q.push_back(vec_active[drive_index]);
                    exp_meta_q.push_back(vec_meta[drive_index]);
                    exp_last_q.push_back(vec_last[drive_index]);
                    sent++;
                    drive_valid = 1'b0;
                end
                cycle++;
                if (cycle > 5000) tb_fail("lane timeout");
            end
            if (exp_result_q.size() != 0) tb_fail("lane expected queue not empty");
        end
    endtask

    initial begin
        load_vectors();
        apply_reset();
        run_vectors();
        $display("STAGE2_PE_LANE_PASS");
        $finish;
    end
endmodule

`default_nettype wire
