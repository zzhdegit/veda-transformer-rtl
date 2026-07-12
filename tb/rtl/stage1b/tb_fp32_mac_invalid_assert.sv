`timescale 1ns/1ps
`default_nettype none

module tb_fp32_mac_invalid_assert;
    logic clk;
    logic rst_n;
    logic in_valid;
    logic in_ready;
    logic out_valid;
    logic out_ready;
    logic [31:0] out_result;
    logic [7:0] out_status;
    logic out_invalid;

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    fp32_mac_wrapper #(
        .META_W(1)
    ) u_dut (
        .clk         (clk),
        .rst_n       (rst_n),
        .in_valid    (in_valid),
        .in_ready    (in_ready),
        .in_a        (32'h7F800000),
        .in_b        (32'h3F800000),
        .in_c        (32'h00000000),
        .in_meta     (1'b0),
        .in_last     (1'b0),
        .out_valid   (out_valid),
        .out_ready   (out_ready),
        .out_result  (out_result),
        .out_status  (out_status),
        .out_invalid (out_invalid),
        .out_meta    (),
        .out_last    ()
    );

    initial begin
        rst_n = 1'b0;
        in_valid = 1'b0;
        out_ready = 1'b1;
        repeat (3) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);
        in_valid = 1'b1;
        @(posedge clk);
        #20;
        $display("STAGE1B_FP32_MAC_INVALID_ASSERT_NOT_TRIGGERED");
        $fatal(1);
    end
endmodule

`default_nettype wire
