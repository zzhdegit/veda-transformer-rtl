`default_nettype none

module pe_lane #(
    parameter int META_W = 1,
    parameter bit ASSERT_ON_INVALID = 1'b1
) (
    input  logic              clk,
    input  logic              rst_n,

    input  logic              in_valid,
    output logic              in_ready,
    input  logic              in_mode,       // 0: product, 1: FMA
    input  logic              in_lane_enable,
    input  logic              in_lane_mask,
    input  logic [31:0]       in_scalar,
    input  logic [31:0]       in_vector,
    input  logic [31:0]       in_accumulator,
    input  logic [META_W-1:0] in_meta,
    input  logic              in_last,

    output logic              out_valid,
    input  logic              out_ready,
    output logic [31:0]       out_result,
    output logic [7:0]        out_status,
    output logic              out_invalid,
    output logic              out_lane_active,
    output logic [META_W-1:0] out_meta,
    output logic              out_last
);
    localparam logic MODE_PRODUCT = 1'b0;
    localparam logic MODE_FMA = 1'b1;

    logic lane_active;
    logic [31:0] mac_a;
    logic [31:0] mac_b;
    logic [31:0] mac_c;
    logic [META_W:0] mac_in_meta;
    logic [META_W:0] mac_out_meta;

    initial begin
        if (META_W <= 0) begin
            $fatal(1, "pe_lane META_W must be positive");
        end
    end

    assign lane_active = in_lane_enable && in_lane_mask;

    always_comb begin
        mac_a = lane_active ? in_scalar : 32'd0;
        mac_b = lane_active ? in_vector : 32'd0;
        mac_c = 32'd0;
        if (in_mode == MODE_FMA) begin
            mac_c = in_accumulator;
        end
    end

    assign mac_in_meta = {lane_active, in_meta};
    assign {out_lane_active, out_meta} = mac_out_meta;

    fp32_mac_wrapper #(
        .META_W(META_W + 1),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_mac (
        .clk         (clk),
        .rst_n       (rst_n),
        .in_valid    (in_valid),
        .in_ready    (in_ready),
        .in_a        (mac_a),
        .in_b        (mac_b),
        .in_c        (mac_c),
        .in_meta     (mac_in_meta),
        .in_last     (in_last),
        .out_valid   (out_valid),
        .out_ready   (out_ready),
        .out_result  (out_result),
        .out_status  (out_status),
        .out_invalid (out_invalid),
        .out_meta    (mac_out_meta),
        .out_last    (out_last)
    );

`ifndef SYNTHESIS
    wire input_fire = in_valid && in_ready;

    always_ff @(posedge clk) begin
        if (rst_n && input_fire) begin
            assert ((in_mode == MODE_PRODUCT) || (in_mode == MODE_FMA))
                else $error("pe_lane unsupported operation mode");
        end
    end
`endif
endmodule

`default_nettype wire
