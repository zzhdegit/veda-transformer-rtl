`timescale 1ns/1ps
`default_nettype none

module tb_fp32_to_fp16;
    localparam int META_W = 16;
    localparam int MAX_CASES = 512;

    logic clk;
    logic rst_n;
    logic in_valid;
    logic in_ready;
    logic [31:0] in_data;
    logic [META_W-1:0] in_meta;
    logic in_last;
    logic out_valid;
    logic out_ready;
    logic [15:0] out_data;
    logic out_invalid;
    logic out_overflow;
    logic out_underflow_or_ftz;
    logic out_inexact;
    logic [META_W-1:0] out_meta;
    logic out_last;

    logic [31:0] vec_in [0:MAX_CASES-1];
    logic [15:0] vec_out [0:MAX_CASES-1];
    logic vec_invalid [0:MAX_CASES-1];
    logic vec_overflow [0:MAX_CASES-1];
    logic vec_ftz [0:MAX_CASES-1];
    logic vec_inexact [0:MAX_CASES-1];
    logic [META_W-1:0] vec_meta [0:MAX_CASES-1];
    logic vec_last [0:MAX_CASES-1];
    int vector_count;

    logic [15:0] exp_data_q[$];
    logic exp_invalid_q[$];
    logic exp_overflow_q[$];
    logic exp_ftz_q[$];
    logic exp_inexact_q[$];
    logic [META_W-1:0] exp_meta_q[$];
    logic exp_last_q[$];

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    fp32_to_fp16 #(
        .META_W(META_W),
        .ASSERT_ON_INVALID(1'b0)
    ) u_dut (
        .clk                  (clk),
        .rst_n                (rst_n),
        .in_valid             (in_valid),
        .in_ready             (in_ready),
        .in_data              (in_data),
        .in_meta              (in_meta),
        .in_last              (in_last),
        .out_valid            (out_valid),
        .out_ready            (out_ready),
        .out_data             (out_data),
        .out_invalid          (out_invalid),
        .out_overflow         (out_overflow),
        .out_underflow_or_ftz (out_underflow_or_ftz),
        .out_inexact          (out_inexact),
        .out_meta             (out_meta),
        .out_last             (out_last)
    );

    task automatic tb_fail(input string message);
        begin
            $display("STAGE6B_FP32_TO_FP16_FAIL: %s", message);
            $fatal(1);
        end
    endtask

    task automatic load_vectors;
        string path;
        int fd;
        int code;
        logic [31:0] in_bits;
        logic [15:0] out_bits;
        int invalid;
        int overflow;
        int ftz;
        int inexact;
        logic [META_W-1:0] meta;
        int last;
        begin
            if (!$value$plusargs("FP32_TO_FP16_VECTOR_FILE=%s", path)) begin
                tb_fail("missing FP32_TO_FP16_VECTOR_FILE");
            end
            fd = $fopen(path, "r");
            if (fd == 0) tb_fail("could not open vector file");
            vector_count = 0;
            while (!$feof(fd)) begin
                code = $fscanf(fd, "%h %h %d %d %d %d %h %d\n",
                               in_bits, out_bits, invalid, overflow, ftz, inexact, meta, last);
                if (code == 8) begin
                    if (vector_count >= MAX_CASES) tb_fail("too many vectors");
                    vec_in[vector_count] = in_bits;
                    vec_out[vector_count] = out_bits;
                    vec_invalid[vector_count] = invalid != 0;
                    vec_overflow[vector_count] = overflow != 0;
                    vec_ftz[vector_count] = ftz != 0;
                    vec_inexact[vector_count] = inexact != 0;
                    vec_meta[vector_count] = meta;
                    vec_last[vector_count] = last != 0;
                    vector_count++;
                end
            end
            $fclose(fd);
            if (vector_count == 0) tb_fail("no vectors loaded");
        end
    endtask

    task automatic apply_reset;
        begin
            rst_n = 1'b0;
            in_valid = 1'b0;
            in_data = '0;
            in_meta = '0;
            in_last = 1'b0;
            out_ready = 1'b0;
            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
            if (out_valid) tb_fail("out_valid not clear after reset");
        end
    endtask

    task automatic pop_and_check(
        input logic [15:0] got_data,
        input logic got_invalid,
        input logic got_overflow,
        input logic got_ftz,
        input logic got_inexact,
        input logic [META_W-1:0] got_meta,
        input logic got_last
    );
        logic [15:0] exp_data;
        logic exp_invalid;
        logic exp_overflow;
        logic exp_ftz;
        logic exp_inexact;
        logic [META_W-1:0] exp_meta;
        logic exp_last;
        begin
            if (exp_data_q.size() == 0) tb_fail("output with empty expected queue");
            exp_data = exp_data_q.pop_front();
            exp_invalid = exp_invalid_q.pop_front();
            exp_overflow = exp_overflow_q.pop_front();
            exp_ftz = exp_ftz_q.pop_front();
            exp_inexact = exp_inexact_q.pop_front();
            exp_meta = exp_meta_q.pop_front();
            exp_last = exp_last_q.pop_front();
            if (got_data !== exp_data) begin
                $display("CHECK_FAIL fp32_to_fp16 data got=%04h expected=%04h meta=%04h",
                         got_data, exp_data, exp_meta);
                $fatal(1);
            end
            if (got_invalid !== exp_invalid) tb_fail("invalid mismatch");
            if (got_overflow !== exp_overflow) tb_fail("overflow mismatch");
            if (got_ftz !== exp_ftz) tb_fail("ftz mismatch");
            if (got_inexact !== exp_inexact) tb_fail("inexact mismatch");
            if (got_meta !== exp_meta) tb_fail("metadata mismatch");
            if (got_last !== exp_last) tb_fail("last mismatch");
        end
    endtask

    task automatic run_stream;
        int sent;
        int received;
        int cycle;
        logic drive_valid;
        int drive_index;
        logic pre_in_fire;
        logic pre_out_fire;
        logic [15:0] pre_out_data;
        logic pre_out_invalid;
        logic pre_out_overflow;
        logic pre_out_ftz;
        logic pre_out_inexact;
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
                if (!drive_valid && sent < vector_count && ((cycle % 5) != 2)) begin
                    drive_valid = 1'b1;
                    drive_index = sent;
                end
                in_valid = drive_valid;
                in_data = vec_in[drive_index];
                in_meta = vec_meta[drive_index];
                in_last = vec_last[drive_index];
                out_ready = ((cycle % 7) != 3) && ((cycle % 11) != 4);
                #1;
                pre_in_fire = in_valid && in_ready;
                pre_out_fire = out_valid && out_ready;
                pre_out_data = out_data;
                pre_out_invalid = out_invalid;
                pre_out_overflow = out_overflow;
                pre_out_ftz = out_underflow_or_ftz;
                pre_out_inexact = out_inexact;
                pre_out_meta = out_meta;
                pre_out_last = out_last;
                @(posedge clk); #1;
                if (pre_out_fire) begin
                    pop_and_check(
                        pre_out_data,
                        pre_out_invalid,
                        pre_out_overflow,
                        pre_out_ftz,
                        pre_out_inexact,
                        pre_out_meta,
                        pre_out_last
                    );
                    received++;
                end
                if (pre_in_fire) begin
                    exp_data_q.push_back(vec_out[drive_index]);
                    exp_invalid_q.push_back(vec_invalid[drive_index]);
                    exp_overflow_q.push_back(vec_overflow[drive_index]);
                    exp_ftz_q.push_back(vec_ftz[drive_index]);
                    exp_inexact_q.push_back(vec_inexact[drive_index]);
                    exp_meta_q.push_back(vec_meta[drive_index]);
                    exp_last_q.push_back(vec_last[drive_index]);
                    sent++;
                    drive_valid = 1'b0;
                end
                cycle++;
                if (cycle > 10000) tb_fail("stream timeout");
            end
            if (exp_data_q.size() != 0) tb_fail("expected queue not empty");
        end
    endtask

    initial begin
        load_vectors();
        apply_reset();
        run_stream();
        $display("STAGE6B_FP32_TO_FP16_PASS cases=%0d", vector_count);
        $finish;
    end
endmodule

`default_nettype wire
