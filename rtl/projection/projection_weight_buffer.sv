`default_nettype none

module projection_weight_buffer #(
    parameter int D_MODEL = 16,
    parameter int MATRIX_KIND_N = 4,
    localparam int KIND_W = (MATRIX_KIND_N <= 1) ? 1 : $clog2(MATRIX_KIND_N),
    localparam int DIM_W = (D_MODEL <= 1) ? 1 : $clog2(D_MODEL)
) (
    input  logic                         clk,
    input  logic                         rst_n,

    input  logic                         clear,

    input  logic                         weight_valid,
    output logic                         weight_ready,
    input  logic [KIND_W-1:0]            weight_kind,
    input  logic [DIM_W-1:0]             weight_output_index,
    input  logic [DIM_W-1:0]             weight_input_index,
    input  logic [15:0]                  weight_data_fp16,
    input  logic                         weight_last,
    input  logic                         weight_commit,

    input  logic [KIND_W-1:0]            read_kind,
    input  logic [DIM_W-1:0]             read_output_index,
    output logic [D_MODEL*16-1:0]        read_row_flat_fp16,
    output logic [D_MODEL-1:0]           read_row_loaded_mask,
    output logic [MATRIX_KIND_N-1:0]     matrix_complete,
    output logic                         error_valid
);
    logic [15:0] data_q [0:MATRIX_KIND_N-1][0:D_MODEL-1][0:D_MODEL-1];
    logic valid_q [0:MATRIX_KIND_N-1][0:D_MODEL-1][0:D_MODEL-1];
    logic error_q;

    wire weight_fire = weight_valid && weight_ready;
    wire kind_legal = int'(weight_kind) < MATRIX_KIND_N;
    wire out_legal = int'(weight_output_index) < D_MODEL;
    wire in_legal = int'(weight_input_index) < D_MODEL;
    wire read_kind_legal = int'(read_kind) < MATRIX_KIND_N;
    wire read_out_legal = int'(read_output_index) < D_MODEL;

    initial begin
        if (D_MODEL <= 0 || MATRIX_KIND_N <= 0) begin
            $fatal(1, "projection_weight_buffer parameters must be positive");
        end
    end

    assign weight_ready = 1'b1;
    assign error_valid = error_q;

    always_comb begin
        read_row_flat_fp16 = '0;
        read_row_loaded_mask = '0;
        matrix_complete = '0;
        for (int kind = 0; kind < MATRIX_KIND_N; kind++) begin
            matrix_complete[kind] = 1'b1;
            for (int out_idx = 0; out_idx < D_MODEL; out_idx++) begin
                for (int in_idx = 0; in_idx < D_MODEL; in_idx++) begin
                    matrix_complete[kind] = matrix_complete[kind] && valid_q[kind][out_idx][in_idx];
                end
            end
        end
        if (read_kind_legal && read_out_legal) begin
            for (int in_idx = 0; in_idx < D_MODEL; in_idx++) begin
                read_row_flat_fp16[in_idx*16 +: 16] = data_q[int'(read_kind)][int'(read_output_index)][in_idx];
                read_row_loaded_mask[in_idx] = valid_q[int'(read_kind)][int'(read_output_index)][in_idx];
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            error_q <= 1'b0;
            for (int kind = 0; kind < MATRIX_KIND_N; kind++) begin
                for (int out_idx = 0; out_idx < D_MODEL; out_idx++) begin
                    for (int in_idx = 0; in_idx < D_MODEL; in_idx++) begin
                        data_q[kind][out_idx][in_idx] <= 16'd0;
                        valid_q[kind][out_idx][in_idx] <= 1'b0;
                    end
                end
            end
        end else begin
            if (clear) begin
                error_q <= 1'b0;
                for (int kind = 0; kind < MATRIX_KIND_N; kind++) begin
                    for (int out_idx = 0; out_idx < D_MODEL; out_idx++) begin
                        for (int in_idx = 0; in_idx < D_MODEL; in_idx++) begin
                            valid_q[kind][out_idx][in_idx] <= 1'b0;
                        end
                    end
                end
            end
            if (weight_fire) begin
                if (!kind_legal || !out_legal || !in_legal) begin
                    error_q <= 1'b1;
                end else begin
                    data_q[int'(weight_kind)][int'(weight_output_index)][int'(weight_input_index)] <= weight_data_fp16;
                    valid_q[int'(weight_kind)][int'(weight_output_index)][int'(weight_input_index)] <= 1'b1;
                end
            end
            if (weight_commit && kind_legal && !matrix_complete[int'(weight_kind)]) begin
                error_q <= 1'b1;
            end
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n) begin
            assert (!(weight_fire && (!kind_legal || !out_legal || !in_legal)))
                else $error("projection_weight_buffer weight_address_in_range failed");
            assert (!(weight_commit && kind_legal && !matrix_complete[int'(weight_kind)]))
                else $error("projection_weight_buffer weight_matrix_complete_before_start failed");
        end
    end
`endif
endmodule

`default_nettype wire
