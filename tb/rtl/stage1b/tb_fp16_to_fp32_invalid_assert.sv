`timescale 1ns/1ps
`default_nettype none

module tb_fp16_to_fp32_invalid_assert;
    logic clk;
    logic rst_n;
    logic in_valid;
    logic in_ready;
    logic [15:0] in_data;
    logic out_valid;
    logic out_ready;
    logic [31:0] out_data;
    logic out_invalid;
    logic out_underflow_or_ftz;

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    fp16_to_fp32 #(
        .META_W(1)
    ) u_dut (
        .clk                  (clk),
        .rst_n                (rst_n),
        .in_valid             (in_valid),
        .in_ready             (in_ready),
        .in_data              (in_data),
        .in_meta              (1'b0),
        .in_last              (1'b0),
        .out_valid            (out_valid),
        .out_ready            (out_ready),
        .out_data             (out_data),
        .out_meta             (),
        .out_last             (),
        .out_invalid          (out_invalid),
        .out_underflow_or_ftz (out_underflow_or_ftz)
    );

    initial begin
        rst_n = 1'b0;
        in_valid = 1'b0;
        in_data = 16'h0000;
        out_ready = 1'b1;
        repeat (3) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);
        in_valid = 1'b1;
        in_data = 16'h7C00;
        @(posedge clk);
        #20;
        $display("STAGE1B_FP16_INVALID_ASSERT_NOT_TRIGGERED");
        $fatal(1);
    end
endmodule

`default_nettype wire
