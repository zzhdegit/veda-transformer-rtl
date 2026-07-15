`timescale 1ns/1ps
`default_nettype none

module tb_ml_m3_numeric_replay;
    localparam int META_W = 1;
    localparam [2:0] WRAPPER_RND_DOCUMENTED = 3'b100;
    localparam int SIG_WIDTH = 23;
    localparam int EXP_WIDTH = 8;
    localparam int IEEE_COMPLIANCE = 1;

    logic clk;
    logic rst_n;
    logic [31:0] a_bits;
    logic [31:0] b_bits;

    logic direct_invalid;
    logic [2:0] direct_rnd;
    logic [31:0] direct_z;
    logic [7:0] direct_status;
    logic [31:0] direct_const_z;
    logic [7:0] direct_const_status;

    logic in_valid;
    logic in_ready;
    logic out_valid;
    logic out_ready;
    logic [31:0] out_result;
    logic [7:0] out_status;
    logic out_invalid;
    logic [META_W-1:0] out_meta;
    logic out_last;

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    assign direct_invalid = (a_bits[30:23] == 8'hFF) || (b_bits[30:23] == 8'hFF);

    DW_fp_add #(
        SIG_WIDTH,
        EXP_WIDTH,
        IEEE_COMPLIANCE
    ) u_direct_add (
        .a      (direct_invalid ? 32'd0 : a_bits),
        .b      (direct_invalid ? 32'd0 : b_bits),
        .rnd    (direct_rnd),
        .z      (direct_z),
        .status (direct_status)
    );

    DW_fp_add #(
        SIG_WIDTH,
        EXP_WIDTH,
        IEEE_COMPLIANCE
    ) u_direct_const_add (
        .a      (direct_invalid ? 32'd0 : a_bits),
        .b      (direct_invalid ? 32'd0 : b_bits),
        .rnd    (WRAPPER_RND_DOCUMENTED),
        .z      (direct_const_z),
        .status (direct_const_status)
    );

    fp32_add_wrapper #(
        .META_W(META_W),
        .ASSERT_ON_INVALID(1'b1)
    ) u_wrapper_add (
        .clk         (clk),
        .rst_n       (rst_n),
        .in_valid    (in_valid),
        .in_ready    (in_ready),
        .in_a        (a_bits),
        .in_b        (b_bits),
        .in_meta     (1'b0),
        .in_last     (1'b1),
        .out_valid   (out_valid),
        .out_ready   (out_ready),
        .out_result  (out_result),
        .out_status  (out_status),
        .out_invalid (out_invalid),
        .out_meta    (out_meta),
        .out_last    (out_last)
    );

    initial begin
        rst_n = 1'b0;
        in_valid = 1'b0;
        out_ready = 1'b0;
        direct_rnd = 3'd0;
        a_bits = 32'h3c81aa0c;
        b_bits = 32'h39699f40;

        void'($value$plusargs("A=%h", a_bits));
        void'($value$plusargs("B=%h", b_bits));

        #1;
        for (int rnd = 0; rnd < 8; rnd++) begin
            direct_rnd = rnd[2:0];
            #20;
            $display("ML_M3_REPLAY_DIRECT_ADD a=%08h b=%08h rnd=%0d result=%08h status=%02h invalid=%0d",
                     a_bits, b_bits, rnd, direct_z, direct_status, direct_invalid);
        end
        #20;
        $display("ML_M3_REPLAY_DIRECT_CONST_ADD a=%08h b=%08h rnd=%0d result=%08h status=%02h invalid=%0d",
                 a_bits, b_bits, WRAPPER_RND_DOCUMENTED, direct_const_z, direct_const_status, direct_invalid);

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        @(negedge clk);
        in_valid = 1'b1;
        @(posedge clk);
        if (!in_ready) begin
            $fatal(1, "numeric replay wrapper was not ready for the single input transaction");
        end
        @(negedge clk);
        in_valid = 1'b0;

        do begin
            @(negedge clk);
        end while (!out_valid);
        #1;
        $display("ML_M3_REPLAY_WRAPPER_INTERNAL rnd_param=%0d dw_a=%08h dw_b=%08h dw_result=%08h dw_status=%02h out_valid=%0d out_ready=%0d",
                 u_wrapper_add.ROUND_NEAREST_EVEN, u_wrapper_add.dw_a, u_wrapper_add.dw_b,
                 u_wrapper_add.dw_result, u_wrapper_add.dw_status, out_valid, out_ready);
        $display("ML_M3_REPLAY_WRAPPER_ADD a=%08h b=%08h documented_rnd=%0d result=%08h status=%02h invalid=%0d",
                 a_bits, b_bits, WRAPPER_RND_DOCUMENTED, out_result, out_status, out_invalid);
        out_ready = 1'b1;
        @(posedge clk);
        $finish;
    end
endmodule

`default_nettype wire
