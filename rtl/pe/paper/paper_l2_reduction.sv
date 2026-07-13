`default_nettype none

module paper_l2_reduction #(
    parameter int META_W = 16,
    parameter bit ASSERT_ON_INVALID = 1'b1
) (
    input  logic              clk,
    input  logic              rst_n,
    input  logic              in_valid,
    output logic              in_ready,
    input  logic [8*32-1:0]   in_values_fp32,
    input  logic [7:0]        in_mask,
    input  logic [META_W-1:0] in_meta,
    input  logic              in_last,
    output logic              out_valid,
    input  logic              out_ready,
    output logic [31:0]       out_sum_fp32,
    output logic [7:0]        out_status,
    output logic              out_invalid,
    output logic [META_W-1:0] out_meta,
    output logic              out_last
);
    paper_l1_reduction #(
        .META_W(META_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_l2_tree (
        .clk            (clk),
        .rst_n          (rst_n),
        .in_valid       (in_valid),
        .in_ready       (in_ready),
        .in_values_fp32 (in_values_fp32),
        .in_mask        (in_mask),
        .in_meta        (in_meta),
        .in_last        (in_last),
        .out_valid      (out_valid),
        .out_ready      (out_ready),
        .out_sum_fp32   (out_sum_fp32),
        .out_status     (out_status),
        .out_invalid    (out_invalid),
        .out_meta       (out_meta),
        .out_last       (out_last)
    );

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n) begin
            assert (!(in_valid && in_ready && $isunknown(in_mask)))
                else $error("paper_l2_reduction l2_reduction_order_legal mask unknown failed");
        end
    end
`endif
endmodule

`default_nettype wire
