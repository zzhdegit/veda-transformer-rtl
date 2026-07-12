`timescale 1ns/1ps
`default_nettype none

module tb_fp32_add_wrapper;
    localparam int META_W = 16;
    localparam int MAX_VECTORS = 256;

    logic clk;
    logic rst_n;
    logic in_valid;
    logic in_ready;
    logic [31:0] in_a;
    logic [31:0] in_b;
    logic [META_W-1:0] in_meta;
    logic in_last;
    logic out_valid;
    logic out_ready;
    logic [31:0] out_result;
    logic [7:0] out_status;
    logic out_invalid;
    logic [META_W-1:0] out_meta;
    logic out_last;

    logic [31:0] vec_a [0:MAX_VECTORS-1];
    logic [31:0] vec_b [0:MAX_VECTORS-1];
    logic [31:0] vec_expected [0:MAX_VECTORS-1];
    logic [META_W-1:0] vec_meta [0:MAX_VECTORS-1];
    logic vec_last [0:MAX_VECTORS-1];
    int vector_count;

    logic [31:0] exp_result_q[$];
    logic [META_W-1:0] exp_meta_q[$];
    logic exp_last_q[$];

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    fp32_add_wrapper #(
        .META_W(META_W)
    ) u_dut (
        .clk         (clk),
        .rst_n       (rst_n),
        .in_valid    (in_valid),
        .in_ready    (in_ready),
        .in_a        (in_a),
        .in_b        (in_b),
        .in_meta     (in_meta),
        .in_last     (in_last),
        .out_valid   (out_valid),
        .out_ready   (out_ready),
        .out_result  (out_result),
        .out_status  (out_status),
        .out_invalid (out_invalid),
        .out_meta    (out_meta),
        .out_last    (out_last)
    );

    task automatic tb_fail(input string message);
        begin
            $display("STAGE2_FP32_ADD_TB_FAIL: %s", message);
            $fatal(1);
        end
    endtask

    task automatic apply_reset;
        begin
            rst_n = 1'b0;
            in_valid = 1'b0;
            in_a = '0;
            in_b = '0;
            in_meta = '0;
            in_last = 1'b0;
            out_ready = 1'b0;
            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
            if (out_valid) tb_fail("out_valid not clear after reset");
        end
    endtask

    task automatic load_vectors;
        string path;
        int fd;
        int code;
        logic [31:0] a;
        logic [31:0] b;
        logic [31:0] expected;
        logic [META_W-1:0] meta;
        logic last;
        begin
            if (!$value$plusargs("ADD_VECTOR_FILE=%s", path)) begin
                tb_fail("missing +ADD_VECTOR_FILE");
            end
            fd = $fopen(path, "r");
            if (fd == 0) tb_fail("could not open add vector file");
            vector_count = 0;
            while (!$feof(fd)) begin
                code = $fscanf(fd, "%h %h %h %h %b\n", a, b, expected, meta, last);
                if (code == 5) begin
                    if (vector_count >= MAX_VECTORS) tb_fail("too many add vectors");
                    vec_a[vector_count] = a;
                    vec_b[vector_count] = b;
                    vec_expected[vector_count] = expected;
                    vec_meta[vector_count] = meta;
                    vec_last[vector_count] = last;
                    vector_count++;
                end
            end
            $fclose(fd);
            if (vector_count == 0) tb_fail("no add vectors loaded");
            $display("STAGE2_FP32_ADD_VECTORS count=%0d", vector_count);
        end
    endtask

    task automatic pop_and_check(
        input logic [31:0] got_result,
        input logic [7:0] got_status,
        input logic got_invalid,
        input logic [META_W-1:0] got_meta,
        input logic got_last
    );
        logic [31:0] expected;
        logic [META_W-1:0] meta;
        logic last;
        begin
            if (exp_result_q.size() == 0) tb_fail("unexpected add output");
            expected = exp_result_q.pop_front();
            meta = exp_meta_q.pop_front();
            last = exp_last_q.pop_front();
            if (got_invalid) tb_fail("unexpected add invalid flag");
            if (^got_status === 1'bx) tb_fail("status contains unknown");
            if (got_result !== expected) begin
                $display("CHECK_FAIL fp32_add result got=%08h expected=%08h meta=%0d status=%02h",
                         got_result, expected, meta, got_status);
                $fatal(1);
            end
            if (got_meta !== meta) tb_fail("metadata mismatch");
            if (got_last !== last) tb_fail("last mismatch");
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
        logic [META_W-1:0] pre_meta;
        logic pre_last;
        begin
            sent = 0;
            received = 0;
            cycle = 0;
            drive_valid = 1'b0;
            drive_index = 0;
            while (received < vector_count) begin
                @(negedge clk);
                if (!drive_valid && (sent < vector_count) && ((cycle % 4) != 1)) begin
                    drive_valid = 1'b1;
                    drive_index = sent;
                end
                in_valid = drive_valid;
                in_a = vec_a[drive_index];
                in_b = vec_b[drive_index];
                in_meta = vec_meta[drive_index];
                in_last = vec_last[drive_index];
                out_ready = ((cycle % 5) != 2) && ((cycle % 13) != 9);
                #1;
                pre_in_fire = in_valid && in_ready;
                pre_out_fire = out_valid && out_ready;
                pre_result = out_result;
                pre_status = out_status;
                pre_invalid = out_invalid;
                pre_meta = out_meta;
                pre_last = out_last;
                @(posedge clk); #1;
                if (pre_out_fire) begin
                    pop_and_check(pre_result, pre_status, pre_invalid, pre_meta, pre_last);
                    received++;
                end
                if (pre_in_fire) begin
                    exp_result_q.push_back(vec_expected[drive_index]);
                    exp_meta_q.push_back(vec_meta[drive_index]);
                    exp_last_q.push_back(vec_last[drive_index]);
                    sent++;
                    drive_valid = 1'b0;
                end
                cycle++;
                if (cycle > 5000) tb_fail("add vector timeout");
            end
            if (exp_result_q.size() != 0) tb_fail("add expected queue not empty");
        end
    endtask

    initial begin
        load_vectors();
        apply_reset();
        run_vectors();
        $display("STAGE2_FP32_ADD_WRAPPER_PASS");
        $finish;
    end
endmodule

`default_nettype wire
