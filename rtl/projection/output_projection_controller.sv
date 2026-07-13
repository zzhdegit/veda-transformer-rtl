`default_nettype none

module output_projection_controller #(
    parameter int D_MODEL = 16,
    parameter int PE_NUM = 8,
    parameter int META_W = 16,
    parameter int COUNTER_W = 64,
    localparam int DIM_W = (D_MODEL <= 1) ? 1 : $clog2(D_MODEL),
    localparam int LEN_W = (D_MODEL <= 1) ? 1 : $clog2(D_MODEL + 1),
    localparam int LANE_W = (PE_NUM <= 1) ? 1 : $clog2(PE_NUM)
) (
    input  logic                         clk,
    input  logic                         rst_n,

    input  logic                         start_valid,
    output logic                         start_ready,
    input  logic [META_W-1:0]            start_meta,
    input  logic [7:0]                   start_status,
    input  logic                         start_invalid,

    output logic                         concat_read_check_valid,
    output logic [DIM_W-1:0]             concat_read_index,
    input  logic [15:0]                  concat_read_data_fp16,
    input  logic                         concat_read_valid,
    input  logic                         concat_complete,
    input  logic                         concat_error,

    output logic                         proj_input_valid,
    input  logic                         proj_input_ready,
    output logic [DIM_W-1:0]             proj_input_dim,
    output logic [15:0]                  proj_input_data_fp16,
    output logic                         proj_input_last,
    output logic                         proj_input_commit,

    output logic                         proj_start_valid,
    input  logic                         proj_start_ready,
    output logic [1:0]                   proj_start_matrix_kind,
    output logic [LEN_W-1:0]             proj_start_input_length,
    output logic [LEN_W-1:0]             proj_start_output_length,
    output logic [META_W-1:0]            proj_start_meta,

    input  logic                         proj_output_valid,
    output logic                         proj_output_ready,
    input  logic [1:0]                   proj_output_matrix_kind,
    input  logic [DIM_W-1:0]             proj_output_index,
    input  logic [31:0]                  proj_output_data_fp32,
    input  logic [7:0]                   proj_output_status,
    input  logic                         proj_output_invalid,
    input  logic [META_W-1:0]            proj_output_meta,
    input  logic                         proj_output_last,

    input  logic                         proj_done_valid,
    output logic                         proj_done_ready,
    input  logic [7:0]                   proj_done_status,
    input  logic                         proj_done_invalid,
    input  logic [META_W-1:0]            proj_done_meta,

    output logic                         output_valid,
    input  logic                         output_ready,
    output logic [DIM_W-1:0]             output_base_dim,
    output logic [PE_NUM*32-1:0]         output_vector_fp32,
    output logic [PE_NUM-1:0]            output_lane_mask,
    output logic [7:0]                   output_status,
    output logic                         output_invalid,
    output logic [META_W-1:0]            output_meta,
    output logic                         output_last,

    output logic                         done_valid,
    input  logic                         done_ready,
    output logic [7:0]                   done_status,
    output logic                         done_invalid,
    output logic [META_W-1:0]            done_meta,

    output logic [COUNTER_W-1:0]         perf_output_projection_cycles,
    output logic [COUNTER_W-1:0]         perf_output_stall_cycles
);
    localparam logic [1:0] KIND_WO = 2'd3;
    localparam logic [7:0] STATUS_CONCAT = 8'hD1;

    typedef enum logic [2:0] {
        ST_IDLE,
        ST_LOAD_INPUT,
        ST_START_PROJ,
        ST_RUN_PROJ,
        ST_DONE
    } state_e;

    state_e state_q;
    logic [DIM_W-1:0] load_index_q;
    logic [META_W-1:0] meta_q;
    logic [7:0] status_q;
    logic invalid_q;
    logic done_seen_q;
    logic final_scalar_seen_q;

    logic out_valid_q;
    logic [DIM_W-1:0] out_base_q;
    logic [PE_NUM*32-1:0] out_vector_q;
    logic [PE_NUM-1:0] out_mask_q;
    logic [7:0] out_status_q;
    logic out_invalid_q;
    logic [META_W-1:0] out_meta_q;
    logic out_last_q;

    logic done_valid_q;
    logic [7:0] done_status_q;
    logic done_invalid_q;
    logic [META_W-1:0] done_meta_q;

    logic [31:0] tile_value_q [0:PE_NUM-1];
    logic [PE_NUM-1:0] tile_mask_q;
    logic [DIM_W-1:0] tile_base_q;
    logic [PE_NUM*32-1:0] tile_vector_with_current;
    logic [PE_NUM-1:0] tile_mask_with_current;

    wire start_fire = start_valid && start_ready;
    wire proj_input_fire = proj_input_valid && proj_input_ready;
    wire proj_start_fire = proj_start_valid && proj_start_ready;
    wire proj_output_fire = proj_output_valid && proj_output_ready;
    wire proj_done_fire = proj_done_valid && proj_done_ready;
    wire output_fire = output_valid && output_ready;
    wire done_fire = done_valid && done_ready;
    wire load_last = load_index_q == DIM_W'(D_MODEL - 1);
    wire [LANE_W-1:0] output_lane = LANE_W'(int'(proj_output_index) % PE_NUM);
    wire output_tile_first_lane = (int'(proj_output_index) % PE_NUM) == 0;
    wire output_tile_last_lane = (int'(proj_output_index) % PE_NUM) == (PE_NUM - 1);
    wire output_final_scalar = proj_output_index == DIM_W'(D_MODEL - 1);
    wire output_tile_complete = output_tile_last_lane || output_final_scalar;

    initial begin
        if (D_MODEL <= 0 || PE_NUM <= 0 || META_W <= 0 || COUNTER_W <= 0) begin
            $fatal(1, "output_projection_controller parameters must be positive");
        end
        if ((PE_NUM & (PE_NUM - 1)) != 0) begin
            $fatal(1, "output_projection_controller PE_NUM must be power of two");
        end
    end

    assign start_ready = (state_q == ST_IDLE) && !done_valid_q && !out_valid_q;

    assign concat_read_index = load_index_q;
    assign concat_read_check_valid = proj_input_fire;

    assign proj_input_valid = (state_q == ST_LOAD_INPUT) && concat_complete && concat_read_valid && !concat_error;
    assign proj_input_dim = load_index_q;
    assign proj_input_data_fp16 = concat_read_data_fp16;
    assign proj_input_last = load_last;
    assign proj_input_commit = proj_input_fire && load_last;

    assign proj_start_valid = state_q == ST_START_PROJ;
    assign proj_start_matrix_kind = KIND_WO;
    assign proj_start_input_length = LEN_W'(D_MODEL);
    assign proj_start_output_length = LEN_W'(D_MODEL);
    assign proj_start_meta = meta_q;
    assign proj_output_ready = (state_q == ST_RUN_PROJ) && !out_valid_q;
    assign proj_done_ready = state_q == ST_RUN_PROJ;

    assign output_valid = out_valid_q;
    assign output_base_dim = out_base_q;
    assign output_vector_fp32 = out_vector_q;
    assign output_lane_mask = out_mask_q;
    assign output_status = out_status_q;
    assign output_invalid = out_invalid_q;
    assign output_meta = out_meta_q;
    assign output_last = out_last_q;

    assign done_valid = done_valid_q;
    assign done_status = done_status_q;
    assign done_invalid = done_invalid_q;
    assign done_meta = done_meta_q;

    always_comb begin
        tile_vector_with_current = '0;
        tile_mask_with_current = tile_mask_q;
        for (int lane = 0; lane < PE_NUM; lane++) begin
            tile_vector_with_current[lane*32 +: 32] = tile_value_q[lane];
        end
        tile_vector_with_current[int'(output_lane)*32 +: 32] = proj_output_data_fp32;
        tile_mask_with_current[int'(output_lane)] = 1'b1;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            load_index_q <= '0;
            meta_q <= '0;
            status_q <= 8'd0;
            invalid_q <= 1'b0;
            done_seen_q <= 1'b0;
            final_scalar_seen_q <= 1'b0;
            out_valid_q <= 1'b0;
            out_base_q <= '0;
            out_vector_q <= '0;
            out_mask_q <= '0;
            out_status_q <= 8'd0;
            out_invalid_q <= 1'b0;
            out_meta_q <= '0;
            out_last_q <= 1'b0;
            done_valid_q <= 1'b0;
            done_status_q <= 8'd0;
            done_invalid_q <= 1'b0;
            done_meta_q <= '0;
            tile_mask_q <= '0;
            tile_base_q <= '0;
            for (int lane = 0; lane < PE_NUM; lane++) begin
                tile_value_q[lane] <= 32'd0;
            end
        end else begin
            if (output_fire) begin
                out_valid_q <= 1'b0;
                out_mask_q <= '0;
                if (out_last_q && done_seen_q) begin
                    done_valid_q <= 1'b1;
                    done_status_q <= status_q;
                    done_invalid_q <= invalid_q;
                    done_meta_q <= meta_q;
                    state_q <= ST_DONE;
                end
            end
            if (done_fire) begin
                done_valid_q <= 1'b0;
                done_status_q <= 8'd0;
                done_invalid_q <= 1'b0;
                done_meta_q <= '0;
                state_q <= ST_IDLE;
            end

            unique case (state_q)
                ST_IDLE: begin
                    if (start_fire) begin
                        load_index_q <= '0;
                        meta_q <= start_meta;
                        status_q <= start_status;
                        invalid_q <= start_invalid;
                        done_seen_q <= 1'b0;
                        final_scalar_seen_q <= 1'b0;
                        tile_mask_q <= '0;
                        tile_base_q <= '0;
                        if (!concat_complete || concat_error) begin
                            done_valid_q <= 1'b1;
                            done_status_q <= start_status | STATUS_CONCAT;
                            done_invalid_q <= 1'b1;
                            done_meta_q <= start_meta;
                            state_q <= ST_DONE;
                        end else begin
                            state_q <= ST_LOAD_INPUT;
                        end
                    end
                end

                ST_LOAD_INPUT: begin
                    if (proj_input_fire) begin
                        if (load_last) begin
                            state_q <= ST_START_PROJ;
                        end else begin
                            load_index_q <= load_index_q + DIM_W'(1);
                        end
                    end
                end

                ST_START_PROJ: begin
                    if (proj_start_fire) begin
                        state_q <= ST_RUN_PROJ;
                    end
                end

                ST_RUN_PROJ: begin
                    if (done_seen_q && final_scalar_seen_q && !out_valid_q && !done_valid_q) begin
                        done_valid_q <= 1'b1;
                        done_status_q <= status_q;
                        done_invalid_q <= invalid_q;
                        done_meta_q <= meta_q;
                        state_q <= ST_DONE;
                    end

                    if (proj_output_fire) begin
                        if (output_tile_first_lane) begin
                            tile_mask_q <= '0;
                            tile_base_q <= proj_output_index;
                        end
                        tile_value_q[int'(output_lane)] <= proj_output_data_fp32;
                        tile_mask_q[int'(output_lane)] <= 1'b1;
                        status_q <= status_q | proj_output_status;
                        invalid_q <= invalid_q | proj_output_invalid;
                        if (output_tile_complete) begin
                            out_valid_q <= 1'b1;
                            out_base_q <= output_tile_first_lane ? proj_output_index : tile_base_q;
                            out_vector_q <= tile_vector_with_current;
                            out_mask_q <= tile_mask_with_current;
                            out_status_q <= status_q | proj_output_status;
                            out_invalid_q <= invalid_q | proj_output_invalid;
                            out_meta_q <= proj_output_meta;
                            out_last_q <= output_final_scalar;
                            tile_mask_q <= '0;
                            final_scalar_seen_q <= output_final_scalar;
                        end
                    end

                    if (proj_done_fire) begin
                        done_seen_q <= 1'b1;
                        status_q <= status_q | proj_done_status;
                        invalid_q <= invalid_q | proj_done_invalid;
                        meta_q <= proj_done_meta;
                        if (final_scalar_seen_q && !out_valid_q) begin
                            done_valid_q <= 1'b1;
                            done_status_q <= status_q | proj_done_status;
                            done_invalid_q <= invalid_q | proj_done_invalid;
                            done_meta_q <= proj_done_meta;
                            state_q <= ST_DONE;
                        end
                    end
                end

                ST_DONE: begin
                    if (!done_valid_q) begin
                        state_q <= ST_IDLE;
                    end
                end

                default: state_q <= ST_IDLE;
            endcase
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            perf_output_projection_cycles <= '0;
            perf_output_stall_cycles <= '0;
        end else begin
            if ((state_q != ST_IDLE) || start_valid) begin
                perf_output_projection_cycles <= perf_output_projection_cycles + COUNTER_W'(1);
            end
            if (output_valid && !output_ready) begin
                perf_output_stall_cycles <= perf_output_stall_cycles + COUNTER_W'(1);
            end
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n) begin
            assert (!(start_fire && (!concat_complete || concat_error)))
                else $error("output_projection_controller no_output_projection_before_concat_complete failed");
            assert (!(proj_start_valid && proj_start_matrix_kind != KIND_WO))
                else $error("output_projection_controller wo_matrix_kind_selection failed");
            assert (!(output_valid && $isunknown({output_base_dim, output_vector_fp32, output_lane_mask,
                                                  output_status, output_invalid, output_meta, output_last})))
                else $error("output_projection_controller no_unknown_output_when_valid failed");
            if ($past(rst_n) && $past(output_valid && !output_ready)) begin
                assert (output_valid)
                    else $error("output_projection_controller output valid dropped under backpressure");
                assert ($stable({output_base_dim, output_vector_fp32, output_lane_mask,
                                 output_status, output_invalid, output_meta, output_last}))
                    else $error("output_projection_controller output_stable_until_ready failed");
            end
        end
    end
`endif
endmodule

`default_nettype wire
