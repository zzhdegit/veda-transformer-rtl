`timescale 1ns/1ps
`default_nettype none

module tb_fp32_mac_wrapper;
    localparam int META_W = 16;
    localparam int MAX_VECTORS = 512;

    logic clk;
    logic rst_n;
    logic in_valid;
    logic in_ready;
    logic [31:0] in_a;
    logic [31:0] in_b;
    logic [31:0] in_c;
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
    logic [31:0] vec_c [0:MAX_VECTORS-1];
    logic [31:0] vec_fused [0:MAX_VECTORS-1];
    logic [31:0] vec_non_fused [0:MAX_VECTORS-1];
    logic [META_W-1:0] vec_meta [0:MAX_VECTORS-1];
    logic vec_last [0:MAX_VECTORS-1];
    int vector_count;
    int expect_fused;

    logic [31:0] exp_result_q[$];
    logic [META_W-1:0] exp_meta_q[$];
    logic exp_last_q[$];

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    fp32_mac_wrapper #(
        .META_W(META_W)
    ) u_dut (
        .clk         (clk),
        .rst_n       (rst_n),
        .in_valid    (in_valid),
        .in_ready    (in_ready),
        .in_a        (in_a),
        .in_b        (in_b),
        .in_c        (in_c),
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
            $display("STAGE1B_FP32_MAC_TB_FAIL: %s", message);
            $fatal(1);
        end
    endtask

    task automatic apply_reset;
        begin
            rst_n = 1'b0;
            in_valid = 1'b0;
            in_a = '0;
            in_b = '0;
            in_c = '0;
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
        logic [31:0] c;
        logic [31:0] fused;
        logic [31:0] non_fused;
        logic [META_W-1:0] meta;
        logic last;
        begin
            if (!$value$plusargs("VECTOR_FILE=%s", path)) begin
                tb_fail("missing +VECTOR_FILE");
            end
            expect_fused = 1;
            void'($value$plusargs("EXPECT_FUSED=%d", expect_fused));
            fd = $fopen(path, "r");
            if (fd == 0) tb_fail("could not open vector file");
            vector_count = 0;
            while (!$feof(fd)) begin
                code = $fscanf(fd, "%h %h %h %h %h %h %b\n",
                               a, b, c, fused, non_fused, meta, last);
                if (code == 7) begin
                    if (vector_count >= MAX_VECTORS) tb_fail("too many vectors");
                    vec_a[vector_count] = a;
                    vec_b[vector_count] = b;
                    vec_c[vector_count] = c;
                    vec_fused[vector_count] = fused;
                    vec_non_fused[vector_count] = non_fused;
                    vec_meta[vector_count] = meta;
                    vec_last[vector_count] = last;
                    vector_count++;
                end
            end
            $fclose(fd);
            if (vector_count == 0) tb_fail("no vectors loaded");
            $display("STAGE1B_FP32_MAC_VECTORS count=%0d expect_fused=%0d", vector_count, expect_fused);
        end
    endtask

    task automatic pop_and_check(
        input logic [31:0] got_result,
        input logic [7:0] got_status,
        input logic got_invalid,
        input logic [META_W-1:0] got_meta,
        input logic got_last
    );
        logic [31:0] exp_result;
        logic [META_W-1:0] exp_meta;
        logic exp_last;
        begin
            if (exp_result_q.size() == 0) tb_fail("output with empty expected queue");
            exp_result = exp_result_q.pop_front();
            exp_meta = exp_meta_q.pop_front();
            exp_last = exp_last_q.pop_front();
            if (got_invalid) tb_fail("unexpected invalid flag");
            if (^got_status === 1'bx) tb_fail("status contains unknown");
            if (got_result !== exp_result) begin
                $display("CHECK_FAIL fp32_mac result got=%08h expected=%08h meta=%0d status=%02h",
                         got_result, exp_result, exp_meta, got_status);
                $fatal(1);
            end
            if (exp_meta == 16'h00F0) begin
                $display("STAGE1B_FP32_MAC_DISCRIMINATOR got=%08h expected=%08h status=%02h",
                         got_result, exp_result, got_status);
            end
            if (got_meta !== exp_meta) tb_fail("metadata mismatch");
            if (got_last !== exp_last) tb_fail("last mismatch");
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
        logic [31:0] pre_out_result;
        logic [7:0] pre_out_status;
        logic pre_out_invalid;
        logic [META_W-1:0] pre_out_meta;
        logic pre_out_last;
        begin
            sent = 0;
            received = 0;
            cycle = 0;
            drive_valid = 1'b0;
            drive_index = 0;

            while (received < vector_count) begin
                @(negedge clk);
                if (!drive_valid && (sent < vector_count) && ((cycle % 5) != 2)) begin
                    drive_valid = 1'b1;
                    drive_index = sent;
                end
                in_valid = drive_valid;
                in_a = vec_a[drive_index];
                in_b = vec_b[drive_index];
                in_c = vec_c[drive_index];
                in_meta = vec_meta[drive_index];
                in_last = vec_last[drive_index];
                out_ready = ((cycle % 7) != 3) && ((cycle % 11) != 8);
                #1;
                pre_in_fire = in_valid && in_ready;
                pre_out_fire = out_valid && out_ready;
                pre_out_result = out_result;
                pre_out_status = out_status;
                pre_out_invalid = out_invalid;
                pre_out_meta = out_meta;
                pre_out_last = out_last;
                @(posedge clk); #1;
                if (pre_out_fire) begin
                    pop_and_check(pre_out_result, pre_out_status, pre_out_invalid, pre_out_meta, pre_out_last);
                    received++;
                end
                if (pre_in_fire) begin
                    exp_result_q.push_back(expect_fused ? vec_fused[drive_index] : vec_non_fused[drive_index]);
                    exp_meta_q.push_back(vec_meta[drive_index]);
                    exp_last_q.push_back(vec_last[drive_index]);
                    sent++;
                    drive_valid = 1'b0;
                end
                cycle++;
                if (cycle > 10000) tb_fail("MAC vector timeout");
            end
            if (sent != vector_count) tb_fail("not all MAC vectors sent");
            if (exp_result_q.size() != 0) tb_fail("MAC expected queue not empty");
        end
    endtask

    initial begin
        load_vectors();
        apply_reset();
        $display("TEST fp32_mac_wrapper");
        run_vectors();
        $display("STAGE1B_FP32_MAC_WRAPPER_PASS");
        $finish;
    end
endmodule

`default_nettype wire
