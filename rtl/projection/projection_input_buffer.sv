`default_nettype none

module projection_input_buffer #(
    parameter int D_MODEL = 16,
    localparam int DIM_W = (D_MODEL <= 1) ? 1 : $clog2(D_MODEL)
) (
    input  logic                     clk,
    input  logic                     rst_n,

    input  logic                     clear,

    input  logic                     load_valid,
    output logic                     load_ready,
    input  logic [DIM_W-1:0]         load_index,
    input  logic [15:0]              load_data_fp16,
    input  logic                     load_last,
    input  logic                     load_commit,

    output logic [D_MODEL*16-1:0]    vector_flat_fp16,
    output logic [D_MODEL-1:0]       loaded_mask,
    output logic                     complete,
    output logic                     error_valid
);
    logic [15:0] data_q [0:D_MODEL-1];
    logic [D_MODEL-1:0] loaded_q;
    logic [D_MODEL-1:0] loaded_after_write;
    logic error_q;

    wire load_fire = load_valid && load_ready;
    wire index_legal = int'(load_index) < D_MODEL;

    initial begin
        if (D_MODEL <= 0) begin
            $fatal(1, "projection_input_buffer D_MODEL must be positive");
        end
    end

    assign load_ready = 1'b1;
    assign loaded_mask = loaded_q;
    assign complete = &loaded_q;
    assign error_valid = error_q;

    always_comb begin
        loaded_after_write = loaded_q;
        if (load_fire && index_legal) begin
            loaded_after_write[int'(load_index)] = 1'b1;
        end
    end

    always_comb begin
        vector_flat_fp16 = '0;
        for (int idx = 0; idx < D_MODEL; idx++) begin
            vector_flat_fp16[idx*16 +: 16] = data_q[idx];
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            loaded_q <= '0;
            error_q <= 1'b0;
            for (int idx = 0; idx < D_MODEL; idx++) begin
                data_q[idx] <= 16'd0;
            end
        end else begin
            if (clear) begin
                loaded_q <= '0;
                error_q <= 1'b0;
            end
            if (load_fire) begin
                if (!index_legal) begin
                    error_q <= 1'b1;
                end else begin
                    data_q[int'(load_index)] <= load_data_fp16;
                    loaded_q[int'(load_index)] <= 1'b1;
                end
            end
            if (load_commit && !(&loaded_after_write)) begin
                error_q <= 1'b1;
            end
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n) begin
            assert (!(load_fire && !index_legal))
                else $error("projection_input_buffer hidden_dimension_order_legal failed");
            assert (!(load_commit && !(&loaded_after_write)))
                else $error("projection_input_buffer no_projection_start_without_complete_input failed");
            assert (!(complete && $isunknown(vector_flat_fp16)))
                else $error("projection_input_buffer no_unknown_output_when_valid failed");
        end
    end
`endif
endmodule

`default_nettype wire
