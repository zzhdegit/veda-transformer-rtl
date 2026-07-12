`default_nettype none

module shared_gemv_projection_core #(
    parameter int D_MODEL = 16,
    parameter int PE_NUM = 8,
    parameter int META_W = 16,
    parameter int COUNTER_W = 64,
    parameter bit ASSERT_ON_INVALID = 1'b1,
    localparam int DIM_W = (D_MODEL <= 1) ? 1 : $clog2(D_MODEL),
    localparam int LEN_W = (D_MODEL <= 1) ? 1 : $clog2(D_MODEL + 1),
    localparam int LANE_COUNT_W = $clog2(PE_NUM + 1)
) (
    input  logic                         clk,
    input  logic                         rst_n,

    input  logic                         command_valid,
    output logic                         command_ready,
    input  logic [1:0]                   command_matrix_kind,
    input  logic [LEN_W-1:0]             command_input_length,
    input  logic [DIM_W-1:0]             command_output_index,
    input  logic [D_MODEL*16-1:0]        command_input_vector_fp16,
    input  logic [D_MODEL*16-1:0]        command_weight_row_fp16,
    input  logic [META_W-1:0]            command_meta,
    input  logic                         command_last,

    output logic                         output_valid,
    input  logic                         output_ready,
    output logic [1:0]                   output_matrix_kind,
    output logic [DIM_W-1:0]             output_index,
    output logic [31:0]                  output_data_fp32,
    output logic [PE_NUM-1:0]            output_lane_mask,
    output logic [7:0]                   output_status,
    output logic                         output_invalid,
    output logic [META_W-1:0]            output_meta,
    output logic                         output_last,

    output logic [COUNTER_W-1:0]         perf_total_cycles,
    output logic [COUNTER_W-1:0]         perf_tile_cycles,
    output logic [COUNTER_W-1:0]         perf_pe_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_output_stall_cycles
);
    localparam logic [1:0] MODE_GEMV = 2'd0;

    typedef enum logic [1:0] {
        ST_IDLE,
        ST_SEND_TILE,
        ST_WAIT_OUTPUT
    } state_e;

    state_e state_q = ST_IDLE;

    logic [1:0] matrix_kind_q;
    logic [LEN_W-1:0] input_length_q;
    logic [DIM_W-1:0] output_index_q;
    logic [D_MODEL*16-1:0] input_vector_q;
    logic [D_MODEL*16-1:0] weight_row_q;
    logic [META_W-1:0] meta_q;
    logic last_q;
    logic [LEN_W-1:0] tile_base_q;

    logic out_valid_q;
    logic [1:0] out_matrix_kind_q;
    logic [DIM_W-1:0] out_index_q;
    logic [31:0] out_data_q;
    logic [PE_NUM-1:0] out_lane_mask_q;
    logic [7:0] out_status_q;
    logic out_invalid_q;
    logic [META_W-1:0] out_meta_q;
    logic out_last_q;

    logic pe_in_valid;
    logic pe_in_ready;
    logic pe_in_tile_first;
    logic pe_in_tile_last;
    logic [LANE_COUNT_W-1:0] pe_in_active_lanes;
    logic [PE_NUM*16-1:0] pe_in_vector_a;
    logic [PE_NUM*16-1:0] pe_in_vector_b;
    logic pe_out_valid;
    logic pe_out_ready;
    logic [1:0] pe_out_mode;
    logic [31:0] pe_out_scalar;
    logic [PE_NUM*32-1:0] pe_out_vector;
    logic [PE_NUM-1:0] pe_out_lane_mask;
    logic [7:0] pe_out_status;
    logic pe_out_invalid;
    logic [META_W-1:0] pe_out_meta;
    logic pe_out_last;

    logic [COUNTER_W-1:0] pe_perf_total_cycles;
    logic [COUNTER_W-1:0] pe_perf_busy_cycles;
    logic [COUNTER_W-1:0] pe_perf_active_lane_cycles;
    logic [COUNTER_W-1:0] pe_perf_available_lane_cycles;
    logic [COUNTER_W-1:0] pe_perf_input_stall_cycles;
    logic [COUNTER_W-1:0] pe_perf_output_stall_cycles;
    logic [COUNTER_W-1:0] pe_perf_mode_switch_cycles;
    logic [COUNTER_W-1:0] pe_perf_tile_count;
    logic [COUNTER_W-1:0] pe_perf_operation_count;
    logic [COUNTER_W-1:0] pe_perf_invalid_count;

    wire command_fire = command_valid && command_ready;
    wire pe_in_fire = pe_in_valid && pe_in_ready;
    wire pe_out_fire = pe_out_valid && pe_out_ready;
    wire output_fire = output_valid && output_ready;
    wire busy = (state_q != ST_IDLE) || out_valid_q;
    wire command_length_legal = (command_input_length > LEN_W'(0)) && (command_input_length <= LEN_W'(D_MODEL));
    wire [LEN_W:0] tile_remaining = {1'b0, input_length_q} - {1'b0, tile_base_q};
    wire [LEN_W:0] tile_width_ext = (tile_remaining > (LEN_W+1)'(PE_NUM)) ? (LEN_W+1)'(PE_NUM) : tile_remaining;

    initial begin
        if (D_MODEL <= 0 || PE_NUM <= 0 || META_W <= 0 || COUNTER_W <= 0) begin
            $fatal(1, "shared_gemv_projection_core parameters must be positive");
        end
        if ((PE_NUM & (PE_NUM - 1)) != 0) begin
            $fatal(1, "shared_gemv_projection_core PE_NUM must be power of two");
        end
    end

    assign command_ready = (state_q == ST_IDLE) && !out_valid_q;
    assign pe_in_valid = state_q == ST_SEND_TILE;
    assign pe_in_tile_first = tile_base_q == LEN_W'(0);
    assign pe_in_tile_last = (tile_base_q + LEN_W'(PE_NUM)) >= input_length_q;
    assign pe_out_ready = !out_valid_q || output_ready;

    assign output_valid = out_valid_q;
    assign output_matrix_kind = out_matrix_kind_q;
    assign output_index = out_index_q;
    assign output_data_fp32 = out_data_q;
    assign output_lane_mask = out_lane_mask_q;
    assign output_status = out_status_q;
    assign output_invalid = out_invalid_q;
    assign output_meta = out_meta_q;
    assign output_last = out_last_q;

    always_comb begin
        pe_in_vector_a = '0;
        pe_in_vector_b = '0;
        pe_in_active_lanes = '0;
        if (state_q == ST_SEND_TILE) begin
            pe_in_active_lanes = LANE_COUNT_W'(tile_width_ext);
        end
        for (int lane = 0; lane < PE_NUM; lane++) begin
            if ((int'(tile_base_q) + lane) < int'(input_length_q)) begin
                pe_in_vector_a[lane*16 +: 16] = input_vector_q[(int'(tile_base_q) + lane)*16 +: 16];
                pe_in_vector_b[lane*16 +: 16] = weight_row_q[(int'(tile_base_q) + lane)*16 +: 16];
            end
        end
    end

    reconfigurable_pe_core #(
        .PE_NUM(PE_NUM),
        .META_W(META_W),
        .COUNTER_W(COUNTER_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_pe_core (
        .clk                         (clk),
        .rst_n                       (rst_n),
        .in_valid                    (pe_in_valid),
        .in_ready                    (pe_in_ready),
        .in_mode                     (MODE_GEMV),
        .in_clear                    (pe_in_tile_first),
        .in_tile_first               (pe_in_tile_first),
        .in_tile_last                (pe_in_tile_last),
        .in_use_explicit_mask        (1'b0),
        .in_active_lanes             (pe_in_active_lanes),
        .in_lane_mask                ('0),
        .in_scalar_fp32              (32'd0),
        .in_vector_a_fp16            (pe_in_vector_a),
        .in_vector_b_fp16            (pe_in_vector_b),
        .in_meta                     (meta_q),
        .in_last                     (last_q),
        .out_valid                   (pe_out_valid),
        .out_ready                   (pe_out_ready),
        .out_mode                    (pe_out_mode),
        .out_scalar_fp32             (pe_out_scalar),
        .out_vector_fp32             (pe_out_vector),
        .out_lane_mask               (pe_out_lane_mask),
        .out_status                  (pe_out_status),
        .out_invalid                 (pe_out_invalid),
        .out_meta                    (pe_out_meta),
        .out_last                    (pe_out_last),
        .perf_total_cycles           (pe_perf_total_cycles),
        .perf_busy_cycles            (pe_perf_busy_cycles),
        .perf_active_lane_cycles     (pe_perf_active_lane_cycles),
        .perf_available_lane_cycles  (pe_perf_available_lane_cycles),
        .perf_input_stall_cycles     (pe_perf_input_stall_cycles),
        .perf_output_stall_cycles    (pe_perf_output_stall_cycles),
        .perf_mode_switch_cycles     (pe_perf_mode_switch_cycles),
        .perf_tile_count             (pe_perf_tile_count),
        .perf_operation_count        (pe_perf_operation_count),
        .perf_invalid_count          (pe_perf_invalid_count)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            matrix_kind_q <= 2'd0;
            input_length_q <= '0;
            output_index_q <= '0;
            input_vector_q <= '0;
            weight_row_q <= '0;
            meta_q <= '0;
            last_q <= 1'b0;
            tile_base_q <= '0;
            out_valid_q <= 1'b0;
            out_matrix_kind_q <= 2'd0;
            out_index_q <= '0;
            out_data_q <= 32'd0;
            out_lane_mask_q <= '0;
            out_status_q <= 8'd0;
            out_invalid_q <= 1'b0;
            out_meta_q <= '0;
            out_last_q <= 1'b0;
        end else begin
            if (output_fire) begin
                out_valid_q <= 1'b0;
            end
            if (pe_out_fire) begin
                out_valid_q <= 1'b1;
                out_matrix_kind_q <= matrix_kind_q;
                out_index_q <= output_index_q;
                out_data_q <= pe_out_scalar;
                out_lane_mask_q <= pe_out_lane_mask;
                out_status_q <= pe_out_status;
                out_invalid_q <= pe_out_invalid;
                out_meta_q <= pe_out_meta;
                out_last_q <= pe_out_last;
            end

            unique case (state_q)
                ST_IDLE: begin
                    if (command_fire) begin
                        matrix_kind_q <= command_matrix_kind;
                        input_length_q <= command_input_length;
                        output_index_q <= command_output_index;
                        input_vector_q <= command_input_vector_fp16;
                        weight_row_q <= command_weight_row_fp16;
                        meta_q <= command_meta;
                        last_q <= command_last;
                        tile_base_q <= '0;
                        state_q <= ST_SEND_TILE;
                    end
                end

                ST_SEND_TILE: begin
                    if (pe_in_fire) begin
                        if (pe_in_tile_last) begin
                            state_q <= ST_WAIT_OUTPUT;
                        end else begin
                            tile_base_q <= tile_base_q + LEN_W'(PE_NUM);
                        end
                    end
                end

                ST_WAIT_OUTPUT: begin
                    if (pe_out_fire) begin
                        state_q <= ST_IDLE;
                    end
                end

                default: state_q <= ST_IDLE;
            endcase
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            perf_total_cycles <= '0;
            perf_tile_cycles <= '0;
            perf_pe_stall_cycles <= '0;
            perf_output_stall_cycles <= '0;
        end else begin
            if (busy || command_valid) begin
                perf_total_cycles <= perf_total_cycles + COUNTER_W'(1);
            end
            if (pe_in_fire) begin
                perf_tile_cycles <= perf_tile_cycles + COUNTER_W'(1);
            end
            if (pe_in_valid && !pe_in_ready) begin
                perf_pe_stall_cycles <= perf_pe_stall_cycles + COUNTER_W'(1);
            end
            if (output_valid && !output_ready) begin
                perf_output_stall_cycles <= perf_output_stall_cycles + COUNTER_W'(1);
            end
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n) begin
            assert (!(command_fire && !command_length_legal))
                else $error("shared_gemv_projection_core input_length_legal failed");
            assert (!(output_valid && $isunknown({output_matrix_kind, output_index, output_data_fp32,
                                                  output_lane_mask, output_status, output_invalid,
                                                  output_meta, output_last})))
                else $error("shared_gemv_projection_core no_unknown_output_when_valid failed");
            if ($past(rst_n) && $past(command_valid && !command_ready)) begin
                assert (command_valid)
                    else $error("shared_gemv_projection_core command valid dropped under backpressure");
                assert ($stable({command_matrix_kind, command_input_length, command_output_index,
                                 command_input_vector_fp16, command_weight_row_fp16,
                                 command_meta, command_last}))
                    else $error("shared_gemv_projection_core command payload changed under backpressure");
            end
            if ($past(rst_n) && $past(output_valid && !output_ready)) begin
                assert (output_valid)
                    else $error("shared_gemv_projection_core output valid dropped under backpressure");
                assert ($stable({output_matrix_kind, output_index, output_data_fp32,
                                 output_lane_mask, output_status, output_invalid,
                                 output_meta, output_last}))
                    else $error("shared_gemv_projection_core output stable until ready failed");
            end
        end
    end
`endif
endmodule

`default_nettype wire
