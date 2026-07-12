`default_nettype none

module fp16_to_fp32 #(
    parameter int META_W = 1,
    parameter bit ASSERT_ON_INVALID = 1'b1
) (
    input  logic              clk,
    input  logic              rst_n,

    input  logic              in_valid,
    output logic              in_ready,
    input  logic [15:0]       in_data,
    input  logic [META_W-1:0] in_meta,
    input  logic              in_last,

    output logic              out_valid,
    input  logic              out_ready,
    output logic [31:0]       out_data,
    output logic [META_W-1:0] out_meta,
    output logic              out_last,
    output logic              out_invalid,
    output logic              out_underflow_or_ftz
);
    localparam int LATENCY = 1;
    localparam int INITIATION_INTERVAL = 1;
    localparam logic [7:0] FP16_TO_FP32_BIAS_DELTA = 8'd112;

    logic        sign;
    logic [4:0]  exp16;
    logic [9:0]  frac16;
    logic        is_zero;
    logic        is_subnormal;
    logic        is_inf_nan;
    logic        is_normal;
    logic [7:0]  exp32;
    logic [31:0] converted_data;
    logic        invalid_comb;
    logic        underflow_or_ftz_comb;
    logic [33:0] pipe_in_data;
    logic [33:0] pipe_out_data;

    initial begin
        if (META_W <= 0) begin
            $fatal(1, "fp16_to_fp32 META_W must be positive");
        end
        if (LATENCY != 1 || INITIATION_INTERVAL != 1) begin
            $fatal(1, "fp16_to_fp32 fixed latency/II parameters changed unexpectedly");
        end
    end

    assign sign = in_data[15];
    assign exp16 = in_data[14:10];
    assign frac16 = in_data[9:0];
    assign is_zero = (exp16 == 5'd0) && (frac16 == 10'd0);
    assign is_subnormal = (exp16 == 5'd0) && (frac16 != 10'd0);
    assign is_inf_nan = (exp16 == 5'h1F);
    assign is_normal = !is_zero && !is_subnormal && !is_inf_nan;
    assign exp32 = {3'b000, exp16} + FP16_TO_FP32_BIAS_DELTA;
    assign invalid_comb = is_inf_nan;
    assign underflow_or_ftz_comb = is_subnormal;

    always_comb begin
        converted_data = {sign, 31'd0};
        if (is_normal) begin
            converted_data = {sign, exp32, frac16, 13'd0};
        end
    end

    assign pipe_in_data = {invalid_comb, underflow_or_ftz_comb, converted_data};
    assign {out_invalid, out_underflow_or_ftz, out_data} = pipe_out_data;

    stream_reg #(
        .DATA_W(34),
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
                else $fatal(1, "fp16_to_fp32 unsupported NaN/Inf input assertion failed");
        end
    end
`endif
endmodule

`default_nettype wire
