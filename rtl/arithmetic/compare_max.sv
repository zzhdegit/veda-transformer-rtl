`default_nettype none

module compare_max #(
    parameter int DATA_W = 32,
    parameter int META_W = 1
) (
    input  logic                    clk,
    input  logic                    rst_n,

    input  logic                    in_valid,
    output logic                    in_ready,
    input  logic signed [DATA_W-1:0] in_a,
    input  logic signed [DATA_W-1:0] in_b,
    input  logic [META_W-1:0]       in_meta,
    input  logic                    in_last,

    output logic                    out_valid,
    input  logic                    out_ready,
    output logic signed [DATA_W-1:0] out_max,
    output logic                    out_take_b,
    output logic [META_W-1:0]       out_meta,
    output logic                    out_last
);
    logic take_b;
    logic signed [DATA_W-1:0] max_value;
    logic [DATA_W:0] pipe_in_data;
    logic [DATA_W:0] pipe_out_data;

    initial begin
        if (DATA_W <= 0 || META_W <= 0) begin
            $fatal(1, "compare_max widths must be positive");
        end
    end

    assign take_b = in_b > in_a;
    assign max_value = take_b ? in_b : in_a;
    assign pipe_in_data = {take_b, max_value};
    assign {out_take_b, out_max} = pipe_out_data;

    stream_reg #(
        .DATA_W(DATA_W + 1),
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
