`default_nettype none

module mac_unit #(
    parameter int A_W = 16,
    parameter int B_W = 16,
    parameter int ACC_W = 40,
    parameter int META_W = 1,
    localparam int PRODUCT_W = A_W + B_W,
    localparam int CALC_W = ((PRODUCT_W > ACC_W) ? PRODUCT_W : ACC_W) + 1
) (
    input  logic                   clk,
    input  logic                   rst_n,

    input  logic                   in_valid,
    output logic                   in_ready,
    input  logic signed [A_W-1:0]  in_a,
    input  logic signed [B_W-1:0]  in_b,
    input  logic signed [ACC_W-1:0] in_acc,
    input  logic                   in_clear,
    input  logic [META_W-1:0]      in_meta,
    input  logic                   in_last,

    output logic                   out_valid,
    input  logic                   out_ready,
    output logic signed [ACC_W-1:0] out_acc,
    output logic                   out_overflow,
    output logic [META_W-1:0]      out_meta,
    output logic                   out_last
);
    logic signed [PRODUCT_W-1:0] product_full;
    logic signed [CALC_W-1:0]    product_ext;
    logic signed [CALC_W-1:0]    acc_ext;
    logic signed [CALC_W-1:0]    mac_full;
    logic signed [ACC_W-1:0]     mac_out;
    logic                        mac_overflow;
    logic [ACC_W:0]              pipe_in_data;
    logic [ACC_W:0]              pipe_out_data;

    initial begin
        if (A_W <= 0 || B_W <= 0 || ACC_W <= 0 || META_W <= 0) begin
            $fatal(1, "mac_unit widths must be positive");
        end
    end

    assign product_full = in_a * in_b;
    assign product_ext  = {{(CALC_W-PRODUCT_W){product_full[PRODUCT_W-1]}}, product_full};
    assign acc_ext      = {{(CALC_W-ACC_W){in_acc[ACC_W-1]}}, in_acc};
    assign mac_full     = in_clear ? product_ext : (acc_ext + product_ext);

    generate
        if (ACC_W == CALC_W) begin : gen_mac_exact
            assign mac_out = mac_full;
            assign mac_overflow = 1'b0;
        end else begin : gen_mac_truncate
            assign mac_out = mac_full[ACC_W-1:0];
            assign mac_overflow =
                mac_full[CALC_W-1:ACC_W] != {(CALC_W-ACC_W){mac_full[ACC_W-1]}};
        end
    endgenerate

    assign pipe_in_data = {mac_overflow, mac_out};
    assign {out_overflow, out_acc} = pipe_out_data;

    stream_reg #(
        .DATA_W(ACC_W + 1),
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
