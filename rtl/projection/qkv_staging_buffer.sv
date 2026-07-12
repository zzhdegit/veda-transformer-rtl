`default_nettype none

module qkv_staging_buffer #(
    parameter int N_HEAD = 2,
    parameter int D_HEAD = 8,
    localparam int D_MODEL = N_HEAD * D_HEAD,
    localparam int HEAD_W = (N_HEAD <= 1) ? 1 : $clog2(N_HEAD),
    localparam int DIM_W = (D_HEAD <= 1) ? 1 : $clog2(D_HEAD),
    localparam int MODEL_W = (D_MODEL <= 1) ? 1 : $clog2(D_MODEL)
) (
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     clear,

    input  logic                     write_valid,
    input  logic [1:0]               write_kind,
    input  logic [MODEL_W-1:0]       write_index,
    input  logic [15:0]              write_data_fp16,

    input  logic [MODEL_W-1:0]       read_index,
    output logic [HEAD_W-1:0]        read_head,
    output logic [DIM_W-1:0]         read_dim,
    output logic [15:0]              read_q_fp16,
    output logic [15:0]              read_k_fp16,
    output logic [15:0]              read_v_fp16,
    output logic                     read_complete,

    output logic [D_MODEL-1:0]       q_loaded_mask,
    output logic [D_MODEL-1:0]       k_loaded_mask,
    output logic [D_MODEL-1:0]       v_loaded_mask,
    output logic                     q_complete,
    output logic                     k_complete,
    output logic                     v_complete,
    output logic                     all_complete,
    output logic                     error_valid
);
    localparam logic [1:0] KIND_Q = 2'd0;
    localparam logic [1:0] KIND_K = 2'd1;
    localparam logic [1:0] KIND_V = 2'd2;

    logic [15:0] q_mem [0:D_MODEL-1];
    logic [15:0] k_mem [0:D_MODEL-1];
    logic [15:0] v_mem [0:D_MODEL-1];
    logic [D_MODEL-1:0] q_loaded_q;
    logic [D_MODEL-1:0] k_loaded_q;
    logic [D_MODEL-1:0] v_loaded_q;
    logic error_q;

    wire index_legal = int'(write_index) < D_MODEL;
    wire read_legal = int'(read_index) < D_MODEL;
    wire kind_legal = (write_kind == KIND_Q) || (write_kind == KIND_K) || (write_kind == KIND_V);

    initial begin
        if (N_HEAD <= 0 || D_HEAD <= 0) begin
            $fatal(1, "qkv_staging_buffer parameters must be positive");
        end
    end

    assign q_loaded_mask = q_loaded_q;
    assign k_loaded_mask = k_loaded_q;
    assign v_loaded_mask = v_loaded_q;
    assign q_complete = &q_loaded_q;
    assign k_complete = &k_loaded_q;
    assign v_complete = &v_loaded_q;
    assign all_complete = q_complete && k_complete && v_complete;
    assign error_valid = error_q;
    assign read_head = HEAD_W'(int'(read_index) / D_HEAD);
    assign read_dim = DIM_W'(int'(read_index) % D_HEAD);
    assign read_q_fp16 = read_legal ? q_mem[int'(read_index)] : 16'd0;
    assign read_k_fp16 = read_legal ? k_mem[int'(read_index)] : 16'd0;
    assign read_v_fp16 = read_legal ? v_mem[int'(read_index)] : 16'd0;
    assign read_complete = read_legal && q_loaded_q[int'(read_index)] &&
                           k_loaded_q[int'(read_index)] && v_loaded_q[int'(read_index)];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q_loaded_q <= '0;
            k_loaded_q <= '0;
            v_loaded_q <= '0;
            error_q <= 1'b0;
            for (int idx = 0; idx < D_MODEL; idx++) begin
                q_mem[idx] <= 16'd0;
                k_mem[idx] <= 16'd0;
                v_mem[idx] <= 16'd0;
            end
        end else begin
            if (clear) begin
                q_loaded_q <= '0;
                k_loaded_q <= '0;
                v_loaded_q <= '0;
                error_q <= 1'b0;
            end
            if (write_valid) begin
                if (!index_legal || !kind_legal) begin
                    error_q <= 1'b1;
                end else begin
                    unique case (write_kind)
                        KIND_Q: begin
                            q_mem[int'(write_index)] <= write_data_fp16;
                            q_loaded_q[int'(write_index)] <= 1'b1;
                        end
                        KIND_K: begin
                            k_mem[int'(write_index)] <= write_data_fp16;
                            k_loaded_q[int'(write_index)] <= 1'b1;
                        end
                        KIND_V: begin
                            v_mem[int'(write_index)] <= write_data_fp16;
                            v_loaded_q[int'(write_index)] <= 1'b1;
                        end
                        default: error_q <= 1'b1;
                    endcase
                end
            end
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n) begin
            assert (!(write_valid && (!index_legal || !kind_legal)))
                else $error("qkv_staging_buffer qkv_output_head_dim_order_legal failed");
            assert (!(all_complete && $isunknown({read_q_fp16, read_k_fp16, read_v_fp16})))
                else $error("qkv_staging_buffer no_unknown_output_when_valid failed");
        end
    end
`endif
endmodule

`default_nettype wire
