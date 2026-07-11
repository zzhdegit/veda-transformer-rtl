`default_nettype none

module mul_unit #(
    parameter int A_W = 16,
    parameter int B_W = 16,
    parameter int OUT_W = A_W + B_W,
    parameter int META_W = 1
) (
    input  logic                  clk,
    input  logic                  rst_n,

    input  logic                  in_valid,
    output logic                  in_ready,
    input  logic signed [A_W-1:0] in_a,
    input  logic signed [B_W-1:0] in_b,
    input  logic [META_W-1:0]     in_meta,
    input  logic                  in_last,

    output logic                  out_valid,
    input  logic                  out_ready,
    output logic signed [OUT_W-1:0] out_result,
    output logic                  out_overflow,
    output logic [META_W-1:0]     out_meta,
    output logic                  out_last
);
    localparam int FULL_W = A_W + B_W;

    logic signed [FULL_W-1:0] product_full;
    logic signed [OUT_W-1:0]  product_out;
    logic                     product_overflow;
    logic [OUT_W:0]           pipe_in_data;
    logic [OUT_W:0]           pipe_out_data;

    initial begin
        if (A_W <= 0 || B_W <= 0 || OUT_W <= 0 || META_W <= 0) begin
            $fatal(1, "mul_unit widths must be positive");
        end
    end

    assign product_full = in_a * in_b;

    generate
        if (OUT_W > FULL_W) begin : gen_mul_extend
            assign product_out = {{(OUT_W-FULL_W){product_full[FULL_W-1]}}, product_full};
            assign product_overflow = 1'b0;
        end else if (OUT_W == FULL_W) begin : gen_mul_exact
            assign product_out = product_full;
            assign product_overflow = 1'b0;
        end else begin : gen_mul_truncate
            assign product_out = product_full[OUT_W-1:0];
            assign product_overflow =
                product_full[FULL_W-1:OUT_W] != {(FULL_W-OUT_W){product_full[OUT_W-1]}};
        end
    endgenerate

    assign pipe_in_data = {product_overflow, product_out};
    assign {out_overflow, out_result} = pipe_out_data;

    stream_reg #(
        .DATA_W(OUT_W + 1),
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
