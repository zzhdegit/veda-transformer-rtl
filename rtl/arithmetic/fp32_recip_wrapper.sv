`default_nettype none

module fp32_recip_wrapper #(
    parameter int META_W = 1,
    parameter int LATENCY = 1,
    parameter int INITIATION_INTERVAL = 1,
    parameter bit ASSERT_ON_INVALID = 1'b1
) (
    input  logic              clk,
    input  logic              rst_n,

    input  logic              in_valid,
    output logic              in_ready,
    input  logic [31:0]       in_a,
    input  logic [META_W-1:0] in_meta,
    input  logic              in_last,

    output logic              out_valid,
    input  logic              out_ready,
    output logic [31:0]       out_result,
    output logic [7:0]        out_status,
    output logic              out_invalid,
    output logic [META_W-1:0] out_meta,
    output logic              out_last
);
    localparam [2:0] ROUND_NEAREST_EVEN = 3'b100;
    localparam int SIG_WIDTH = 23;
    localparam int EXP_WIDTH = 8;
    localparam int IEEE_COMPLIANCE = 1;
    localparam int FAITHFUL_ROUND = 0;
    localparam logic [31:0] FP32_ONE = 32'h3F80_0000;
    localparam logic [31:0] FP32_ZERO = 32'h0000_0000;

    logic a_nonfinite;
    logic a_zero;
    logic invalid_comb;
    logic [31:0] dw_den;
    logic [31:0] dw_result;
    logic [7:0]  dw_status;
    logic [40:0] pipe_in_data;
    logic [40:0] pipe_out_data;

    initial begin
        if (META_W <= 0) begin
            $fatal(1, "fp32_recip_wrapper META_W must be positive");
        end
        if (LATENCY != 1) begin
            $fatal(1, "fp32_recip_wrapper currently exposes exactly one output register stage");
        end
        if (INITIATION_INTERVAL != 1) begin
            $fatal(1, "fp32_recip_wrapper currently supports initiation interval 1");
        end
    end

    assign a_nonfinite = (in_a[30:23] == 8'hFF);
    assign a_zero = (in_a[30:0] == 31'd0);
    assign invalid_comb = a_nonfinite || a_zero;
    assign dw_den = invalid_comb ? FP32_ONE : in_a;

    DW_fp_div #(
        SIG_WIDTH,
        EXP_WIDTH,
        IEEE_COMPLIANCE,
        FAITHFUL_ROUND
    ) u_dw_fp_div (
        .a      (FP32_ONE),
        .b      (dw_den),
        .rnd    (ROUND_NEAREST_EVEN),
        .z      (dw_result),
        .status (dw_status)
    );

    assign pipe_in_data = {
        invalid_comb,
        invalid_comb ? 8'd0 : dw_status,
        invalid_comb ? FP32_ZERO : dw_result
    };
    assign {out_invalid, out_status, out_result} = pipe_out_data;

    stream_reg #(
        .DATA_W(41),
        .META_W(META_W)
    ) u_output_reg (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_valid  (in_valid),
        .in_ready  (in_ready),
        .in_data   (pipe_in_data),
        .in_meta   (in_meta),
        .in_last   (in_last),
        .out_valid (out_valid),
        .out_ready (out_ready),
        .out_data  (pipe_out_data),
        .out_meta  (out_meta),
        .out_last  (out_last)
    );

`ifndef SYNTHESIS
    wire input_fire = in_valid && in_ready;

    always_ff @(posedge clk) begin
        if (rst_n && ASSERT_ON_INVALID && input_fire) begin
            assert (!invalid_comb)
                else $fatal(1, "fp32_recip_wrapper unsupported NaN/Inf/zero input assertion failed");
        end
    end
`endif
endmodule

`default_nettype wire

