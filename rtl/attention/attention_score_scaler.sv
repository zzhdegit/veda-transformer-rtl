`default_nettype none

module attention_score_scaler #(
    parameter int D_HEAD = 8,
    parameter int META_W = 1,
    parameter bit ASSERT_ON_INVALID = 1'b1
) (
    input  logic              clk,
    input  logic              rst_n,

    input  logic              in_valid,
    output logic              in_ready,
    input  logic [31:0]       in_score,
    input  logic [META_W-1:0] in_meta,
    input  logic              in_last,

    output logic              out_valid,
    input  logic              out_ready,
    output logic [31:0]       out_score,
    output logic [7:0]        out_status,
    output logic              out_invalid,
    output logic [META_W-1:0] out_meta,
    output logic              out_last
);
    localparam logic [31:0] FP32_ZERO = 32'h0000_0000;

    function automatic logic [31:0] scale_constant(input int d_head);
        begin
            unique case (d_head)
                1:   scale_constant = 32'h3F80_0000;
                7:   scale_constant = 32'h3EC1_848F;
                8:   scale_constant = 32'h3EB5_04F3;
                9:   scale_constant = 32'h3EAA_AAAB;
                13:  scale_constant = 32'h3E8E_00D5;
                16:  scale_constant = 32'h3E80_0000;
                128: scale_constant = 32'h3DB5_04F3;
                default: scale_constant = 32'h0000_0000;
            endcase
        end
    endfunction

    initial begin
        if (META_W <= 0) begin
            $fatal(1, "attention_score_scaler META_W must be positive");
        end
        if (scale_constant(D_HEAD) == 32'd0) begin
            $fatal(1, "attention_score_scaler unsupported D_HEAD scale constant");
        end
    end

    fp32_mac_wrapper #(
        .META_W(META_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_scale_mac (
        .clk         (clk),
        .rst_n       (rst_n),
        .in_valid    (in_valid),
        .in_ready    (in_ready),
        .in_a        (in_score),
        .in_b        (scale_constant(D_HEAD)),
        .in_c        (FP32_ZERO),
        .in_meta     (in_meta),
        .in_last     (in_last),
        .out_valid   (out_valid),
        .out_ready   (out_ready),
        .out_result  (out_score),
        .out_status  (out_status),
        .out_invalid (out_invalid),
        .out_meta    (out_meta),
        .out_last    (out_last)
    );
endmodule

`default_nettype wire

