`timescale 1ns/1ps
`default_nettype none

module tb_fp16_to_fp32;
    localparam int META_W = 16;
    localparam int TOTAL = 65536;

    logic clk;
    logic rst_n;
    logic in_valid;
    logic in_ready;
    logic [15:0] in_data;
    logic [META_W-1:0] in_meta;
    logic in_last;
    logic out_valid;
    logic out_ready;
    logic [31:0] out_data;
    logic [META_W-1:0] out_meta;
    logic out_last;
    logic out_invalid;
    logic out_underflow_or_ftz;

    logic [31:0] exp_data_q[$];
    logic [META_W-1:0] exp_meta_q[$];
    logic exp_last_q[$];
    logic exp_invalid_q[$];
    logic exp_ftz_q[$];

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    fp16_to_fp32 #(
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
        .out_meta             (out_meta),
        .out_last             (out_last),
        .out_invalid          (out_invalid),
        .out_underflow_or_ftz (out_underflow_or_ftz)
    );

    task automatic tb_fail(input string message);
        begin
            $display("STAGE1B_FP16_TB_FAIL: %s", message);
            $fatal(1);
        end
    endtask

    function automatic [33:0] expected_fp16_to_fp32(input logic [15:0] value);
        logic sign;
        logic [4:0] exp16;
        logic [9:0] frac16;
        logic [7:0] exp32;
        logic [31:0] data32;
        logic invalid;
        logic ftz;
        begin
            sign = value[15];
            exp16 = value[14:10];
            frac16 = value[9:0];
            invalid = (exp16 == 5'h1F);
            ftz = (exp16 == 5'd0) && (frac16 != 10'd0);
            data32 = {sign, 31'd0};
            if ((exp16 != 5'd0) && (exp16 != 5'h1F)) begin
                exp32 = {3'b000, exp16} + 8'd112;
                data32 = {sign, exp32, frac16, 13'd0};
            end
            expected_fp16_to_fp32 = {invalid, ftz, data32};
        end
    endfunction

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
        input logic [31:0] got_data,
        input logic [META_W-1:0] got_meta,
        input logic got_last,
        input logic got_invalid,
        input logic got_ftz
    );
        logic [31:0] exp_data;
        logic [META_W-1:0] exp_meta;
        logic exp_last;
        logic exp_invalid;
        logic exp_ftz;
        begin
            if (exp_data_q.size() == 0) tb_fail("output with empty expected queue");
            exp_data = exp_data_q.pop_front();
            exp_meta = exp_meta_q.pop_front();
            exp_last = exp_last_q.pop_front();
            exp_invalid = exp_invalid_q.pop_front();
            exp_ftz = exp_ftz_q.pop_front();
            if (got_data !== exp_data) begin
                $display("CHECK_FAIL fp16_to_fp32 data got=%08h expected=%08h meta=%0d",
                         got_data, exp_data, exp_meta);
                $fatal(1);
            end
            if (got_meta !== exp_meta) tb_fail("metadata mismatch");
            if (got_last !== exp_last) tb_fail("last mismatch");
            if (got_invalid !== exp_invalid) tb_fail("invalid flag mismatch");
            if (got_ftz !== exp_ftz) tb_fail("ftz flag mismatch");
        end
    endtask

    task automatic test_exhaustive;
        int sent;
        int received;
        int cycle;
        logic drive_valid;
        logic [15:0] drive_value;
        logic pre_in_fire;
        logic pre_out_fire;
        logic [31:0] pre_out_data;
        logic [META_W-1:0] pre_out_meta;
        logic pre_out_last;
        logic pre_out_invalid;
        logic pre_out_ftz;
        logic [33:0] expected;
        begin
            sent = 0;
            received = 0;
            cycle = 0;
            drive_valid = 1'b0;
            drive_value = '0;

            while (received < TOTAL) begin
                @(negedge clk);
                if (!drive_valid && (sent < TOTAL) && ((cycle % 7) != 3)) begin
                    drive_valid = 1'b1;
                    drive_value = sent[15:0];
                end
                in_valid = drive_valid;
                in_data = drive_value;
                in_meta = drive_value ^ 16'hA55A;
                in_last = (drive_value == 16'hFFFF);
                out_ready = ((cycle % 11) != 5) && ((cycle % 13) != 9);
                #1;
                pre_in_fire = in_valid && in_ready;
                pre_out_fire = out_valid && out_ready;
                pre_out_data = out_data;
                pre_out_meta = out_meta;
                pre_out_last = out_last;
                pre_out_invalid = out_invalid;
                pre_out_ftz = out_underflow_or_ftz;
                @(posedge clk); #1;
                if (pre_out_fire) begin
                    pop_and_check(pre_out_data, pre_out_meta, pre_out_last, pre_out_invalid, pre_out_ftz);
                    received++;
                end
                if (pre_in_fire) begin
                    expected = expected_fp16_to_fp32(drive_value);
                    exp_data_q.push_back(expected[31:0]);
                    exp_ftz_q.push_back(expected[32]);
                    exp_invalid_q.push_back(expected[33]);
                    exp_meta_q.push_back(drive_value ^ 16'hA55A);
                    exp_last_q.push_back(drive_value == 16'hFFFF);
                    sent++;
                    drive_valid = 1'b0;
                end
                cycle++;
                if (cycle > 200000) tb_fail("exhaustive timeout");
            end

            if (sent != TOTAL) tb_fail("not all inputs sent");
            if (exp_data_q.size() != 0) tb_fail("expected queue not empty after drain");
        end
    endtask

    initial begin
        apply_reset();
        $display("TEST fp16_to_fp32 exhaustive");
        test_exhaustive();
        $display("STAGE1B_FP16_TO_FP32_PASS");
        $finish;
    end
endmodule

`default_nettype wire
