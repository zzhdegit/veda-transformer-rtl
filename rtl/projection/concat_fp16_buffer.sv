`default_nettype none

module concat_fp16_buffer #(
    parameter int D_MODEL = 16,
    localparam int DIM_W = (D_MODEL <= 1) ? 1 : $clog2(D_MODEL)
) (
    input  logic                  clk,
    input  logic                  rst_n,

    input  logic                  clear,

    input  logic                  write_valid,
    output logic                  write_ready,
    input  logic [DIM_W-1:0]      write_index,
    input  logic [15:0]           write_data_fp16,

    input  logic                  read_check_valid,
    input  logic [DIM_W-1:0]      read_index,
    output logic [15:0]           read_data_fp16,
    output logic                  read_valid,

    input  logic                  complete_check,

    output logic [D_MODEL*16-1:0] vector_flat_fp16,
    output logic [D_MODEL-1:0]    loaded_mask,
    output logic                  complete,
    output logic                  error_valid,
    output logic                  duplicate_error,
    output logic                  missing_error,
    output logic                  range_error
);
    logic [15:0] data_q [0:D_MODEL-1];
    logic [D_MODEL-1:0] loaded_q;
    logic duplicate_error_q;
    logic missing_error_q;
    logic range_error_q;

    wire write_fire = write_valid && write_ready;
    wire write_index_legal = int'(write_index) < D_MODEL;
    wire read_index_legal = int'(read_index) < D_MODEL;

    initial begin
        if (D_MODEL <= 0) begin
            $fatal(1, "concat_fp16_buffer D_MODEL must be positive");
        end
    end

    assign write_ready = 1'b1;
    assign loaded_mask = loaded_q;
    assign complete = &loaded_q;
    assign duplicate_error = duplicate_error_q;
    assign missing_error = missing_error_q;
    assign range_error = range_error_q;
    assign error_valid = duplicate_error_q || missing_error_q || range_error_q;
    assign read_valid = read_index_legal && loaded_q[int'(read_index)];
    assign read_data_fp16 = read_index_legal ? data_q[int'(read_index)] : 16'd0;

    always_comb begin
        vector_flat_fp16 = '0;
        for (int idx = 0; idx < D_MODEL; idx++) begin
            vector_flat_fp16[idx*16 +: 16] = data_q[idx];
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            loaded_q <= '0;
            duplicate_error_q <= 1'b0;
            missing_error_q <= 1'b0;
            range_error_q <= 1'b0;
            for (int idx = 0; idx < D_MODEL; idx++) begin
                data_q[idx] <= 16'd0;
            end
        end else begin
            if (clear) begin
                loaded_q <= '0;
                duplicate_error_q <= 1'b0;
                missing_error_q <= 1'b0;
                range_error_q <= 1'b0;
            end

            if (write_fire) begin
                if (!write_index_legal) begin
                    range_error_q <= 1'b1;
                end else begin
                    if (loaded_q[int'(write_index)]) begin
                        duplicate_error_q <= 1'b1;
                    end
                    data_q[int'(write_index)] <= write_data_fp16;
                    loaded_q[int'(write_index)] <= 1'b1;
                end
            end

            if (read_check_valid && (!read_index_legal || !loaded_q[int'(read_index)])) begin
                range_error_q <= range_error_q || !read_index_legal;
                missing_error_q <= missing_error_q || read_index_legal;
            end
            if (complete_check && !(&loaded_q)) begin
                missing_error_q <= 1'b1;
            end
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n) begin
            assert (!(write_fire && !write_index_legal))
                else $error("concat_fp16_buffer concat_index_in_range failed");
            assert (!(write_fire && write_index_legal && loaded_q[int'(write_index)]))
                else $error("concat_fp16_buffer no_duplicate_concat_write failed");
            assert (!(complete_check && !complete))
                else $error("concat_fp16_buffer concat_complete_requires_all_writes failed");
            assert (!(read_check_valid && (!read_index_legal || !loaded_q[int'(read_index)])))
                else $error("concat_fp16_buffer no_output_projection_before_concat_complete failed");
        end
    end
`endif
endmodule

`default_nettype wire
