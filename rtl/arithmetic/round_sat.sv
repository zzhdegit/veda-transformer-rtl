`default_nettype none

module round_sat #(
    parameter int IN_W = 32,
    parameter int OUT_W = 16,
    parameter int FRAC_DROP = 0,
    parameter int ROUND_MODE = 0,
    parameter int SATURATE = 1,
    parameter int META_W = 1,
    localparam int CALC_W = IN_W + 2
) (
    input  logic                     clk,
    input  logic                     rst_n,

    input  logic                     in_valid,
    output logic                     in_ready,
    input  logic signed [IN_W-1:0]   in_data,
    input  logic [META_W-1:0]        in_meta,
    input  logic                     in_last,

    output logic                     out_valid,
    input  logic                     out_ready,
    output logic signed [OUT_W-1:0]  out_data,
    output logic                     out_overflow,
    output logic                     out_underflow,
    output logic                     out_inexact,
    output logic [META_W-1:0]        out_meta,
    output logic                     out_last
);
    localparam int ROUND_TRUNCATE = 0;
    localparam int ROUND_NEAREST_EVEN = 1;

    logic signed [CALC_W-1:0] rounded_ext;
    logic                     inexact_comb;
    logic signed [CALC_W-1:0] sat_max;
    logic signed [CALC_W-1:0] sat_min;
    logic signed [OUT_W-1:0]  sat_data;
    logic                     overflow_comb;
    logic                     underflow_comb;
    logic [OUT_W+2:0]         pipe_in_data;
    logic [OUT_W+2:0]         pipe_out_data;

    initial begin
        if (IN_W <= 0 || OUT_W <= 0 || META_W <= 0) begin
            $fatal(1, "round_sat widths must be positive");
        end
        if (OUT_W >= CALC_W) begin
            $fatal(1, "round_sat OUT_W must be narrower than internal calculation width");
        end
        if (FRAC_DROP < 0 || FRAC_DROP >= IN_W) begin
            $fatal(1, "round_sat FRAC_DROP must be in range [0, IN_W)");
        end
        if (ROUND_MODE != ROUND_TRUNCATE && ROUND_MODE != ROUND_NEAREST_EVEN) begin
            $fatal(1, "round_sat unsupported ROUND_MODE");
        end
    end

    assign sat_max = {{(CALC_W-OUT_W){1'b0}}, 1'b0, {(OUT_W-1){1'b1}}};
    assign sat_min = {{(CALC_W-OUT_W){1'b1}}, 1'b1, {(OUT_W-1){1'b0}}};

    generate
        if (FRAC_DROP == 0) begin : gen_no_round
            assign rounded_ext = {{(CALC_W-IN_W){in_data[IN_W-1]}}, in_data};
            assign inexact_comb = 1'b0;
        end else if (ROUND_MODE == ROUND_TRUNCATE) begin : gen_truncate
            logic signed [IN_W-1:0] shifted;
            assign shifted = in_data >>> FRAC_DROP;
            assign rounded_ext = {{(CALC_W-IN_W){shifted[IN_W-1]}}, shifted};
            assign inexact_comb = |in_data[FRAC_DROP-1:0];
        end else begin : gen_rne
            logic [IN_W:0] abs_mag;
            logic [IN_W:0] abs_trunc;
            logic [IN_W:0] input_ext_bits;
            logic [FRAC_DROP-1:0] remainder;
            logic [FRAC_DROP-1:0] half_value;
            logic round_inc;
            logic [IN_W:0] rounded_mag;
            logic signed [CALC_W-1:0] positive_ext;

            assign input_ext_bits = {in_data[IN_W-1], in_data};
            assign abs_mag = in_data[IN_W-1] ? (~input_ext_bits + {{IN_W{1'b0}}, 1'b1}) :
                                                input_ext_bits;
            assign abs_trunc = abs_mag >> FRAC_DROP;
            assign remainder = abs_mag[FRAC_DROP-1:0];
            assign half_value = {1'b1, {(FRAC_DROP-1){1'b0}}};
            assign round_inc = (remainder > half_value) ||
                               ((remainder == half_value) && abs_trunc[0]);
            assign rounded_mag = abs_trunc + {{IN_W{1'b0}}, round_inc};
            assign positive_ext = {{(CALC_W-(IN_W+1)){1'b0}}, rounded_mag};
            assign rounded_ext = in_data[IN_W-1] ? -positive_ext : positive_ext;
            assign inexact_comb = |remainder;
        end
    endgenerate

    always_comb begin
        overflow_comb = 1'b0;
        underflow_comb = 1'b0;
        sat_data = rounded_ext[OUT_W-1:0];

        if (SATURATE != 0) begin
            if (rounded_ext > sat_max) begin
                sat_data = sat_max[OUT_W-1:0];
                overflow_comb = 1'b1;
            end else if (rounded_ext < sat_min) begin
                sat_data = sat_min[OUT_W-1:0];
                underflow_comb = 1'b1;
            end
        end else begin
            overflow_comb =
                rounded_ext[CALC_W-1:OUT_W] != {(CALC_W-OUT_W){rounded_ext[OUT_W-1]}};
            underflow_comb = overflow_comb && rounded_ext[CALC_W-1];
        end
    end

    assign pipe_in_data = {overflow_comb, underflow_comb, inexact_comb, sat_data};
    assign {out_overflow, out_underflow, out_inexact, out_data} = pipe_out_data;

    stream_reg #(
        .DATA_W(OUT_W + 3),
        .META_W(META_W)
    ) u_result_reg (
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
endmodule

`default_nettype wire
