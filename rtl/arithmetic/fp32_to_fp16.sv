`default_nettype none

module fp32_to_fp16 #(
    parameter int META_W = 1,
    parameter bit ASSERT_ON_INVALID = 1'b1
) (
    input  logic              clk,
    input  logic              rst_n,

    input  logic              in_valid,
    output logic              in_ready,
    input  logic [31:0]       in_data,
    input  logic [META_W-1:0] in_meta,
    input  logic              in_last,

    output logic              out_valid,
    input  logic              out_ready,
    output logic [15:0]       out_data,
    output logic              out_invalid,
    output logic              out_overflow,
    output logic              out_underflow_or_ftz,
    output logic              out_inexact,
    output logic [META_W-1:0] out_meta,
    output logic              out_last
);
    localparam int LATENCY = 1;
    localparam int INITIATION_INTERVAL = 1;

    logic sign;
    logic [7:0] exp32;
    logic [22:0] frac32;
    logic is_zero;
    logic is_subnormal;
    logic is_inf_nan;
    logic is_normal;
    logic signed [9:0] exponent_unbiased;
    logic signed [9:0] exponent16_signed;
    logic [23:0] significand;
    logic [10:0] significand_main;
    logic guard_bit;
    logic sticky_bit;
    logic round_increment;
    logic [11:0] rounded_significand;
    logic signed [9:0] rounded_exponent16;
    logic [15:0] converted_data;
    logic invalid_comb;
    logic overflow_comb;
    logic underflow_or_ftz_comb;
    logic inexact_comb;
    logic [19:0] pipe_in_data;
    logic [19:0] pipe_out_data;

    initial begin
        if (META_W <= 0) begin
            $fatal(1, "fp32_to_fp16 META_W must be positive");
        end
        if (LATENCY != 1 || INITIATION_INTERVAL != 1) begin
            $fatal(1, "fp32_to_fp16 fixed latency/II parameters changed unexpectedly");
        end
    end

    assign sign = in_data[31];
    assign exp32 = in_data[30:23];
    assign frac32 = in_data[22:0];
    assign is_zero = (exp32 == 8'd0) && (frac32 == 23'd0);
    assign is_subnormal = (exp32 == 8'd0) && (frac32 != 23'd0);
    assign is_inf_nan = (exp32 == 8'hFF);
    assign is_normal = !is_zero && !is_subnormal && !is_inf_nan;
    assign exponent_unbiased = $signed({2'b00, exp32}) - 10'sd127;
    assign exponent16_signed = exponent_unbiased + 10'sd15;
    assign significand = {1'b1, frac32};
    assign significand_main = significand[23:13];
    assign guard_bit = significand[12];
    assign sticky_bit = |significand[11:0];
    assign round_increment = guard_bit && (sticky_bit || significand_main[0]);
    assign rounded_significand = {1'b0, significand_main} + 12'(round_increment);
    assign rounded_exponent16 = exponent16_signed + (rounded_significand[11] ? 10'sd1 : 10'sd0);

    always_comb begin
        converted_data = {sign, 15'd0};
        invalid_comb = 1'b0;
        overflow_comb = 1'b0;
        underflow_or_ftz_comb = 1'b0;
        inexact_comb = 1'b0;

        if (is_inf_nan) begin
            invalid_comb = 1'b1;
            inexact_comb = 1'b1;
            converted_data = {sign, 15'd0};
        end else if (is_zero) begin
            converted_data = {sign, 15'd0};
        end else if (is_subnormal) begin
            underflow_or_ftz_comb = 1'b1;
            inexact_comb = 1'b1;
            converted_data = {sign, 15'd0};
        end else if (is_normal && (exponent_unbiased < -10'sd14)) begin
            underflow_or_ftz_comb = 1'b1;
            inexact_comb = 1'b1;
            converted_data = {sign, 15'd0};
        end else if (is_normal && ((exponent_unbiased > 10'sd15) || (rounded_exponent16 >= 10'sd31))) begin
            overflow_comb = 1'b1;
            inexact_comb = 1'b1;
            converted_data = {sign, 5'h1E, 10'h3FF};
        end else if (is_normal) begin
            converted_data = {
                sign,
                rounded_exponent16[4:0],
                (rounded_significand[11] ? rounded_significand[10:1] : rounded_significand[9:0])
            };
            inexact_comb = guard_bit || sticky_bit;
        end
    end

    assign pipe_in_data = {invalid_comb, overflow_comb, underflow_or_ftz_comb, inexact_comb, converted_data};
    assign {out_invalid, out_overflow, out_underflow_or_ftz, out_inexact, out_data} = pipe_out_data;

    stream_reg #(
        .DATA_W(20),
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
                else $fatal(1, "fp32_to_fp16 unsupported NaN/Inf input assertion failed");
        end
        if (rst_n) begin
            if ($past(rst_n) && $past(out_valid && !out_ready)) begin
                assert (out_valid)
                    else $error("fp32_to_fp16 output valid dropped under backpressure");
                assert ($stable({out_data, out_invalid, out_overflow, out_underflow_or_ftz,
                                 out_inexact, out_meta, out_last}))
                    else $error("fp32_to_fp16 output stable until ready failed");
            end
        end
    end
`endif
endmodule

`default_nettype wire
