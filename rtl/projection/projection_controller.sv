`default_nettype none

module projection_controller #(
    parameter int D_MODEL = 16,
    parameter int PE_NUM = 8,
    parameter int META_W = 16,
    parameter int COUNTER_W = 64,
    parameter bit ASSERT_ON_INVALID = 1'b1,
    localparam int DIM_W = (D_MODEL <= 1) ? 1 : $clog2(D_MODEL),
    localparam int LEN_W = (D_MODEL <= 1) ? 1 : $clog2(D_MODEL + 1)
) (
    input  logic                         clk,
    input  logic                         rst_n,

    input  logic                         input_valid,
    output logic                         input_ready,
    input  logic [DIM_W-1:0]             input_dim,
    input  logic [15:0]                  input_data_fp16,
    input  logic                         input_last,
    input  logic                         input_commit,

    input  logic                         weight_valid,
    output logic                         weight_ready,
    input  logic [1:0]                   weight_kind,
    input  logic [DIM_W-1:0]             weight_output_index,
    input  logic [DIM_W-1:0]             weight_input_index,
    input  logic [15:0]                  weight_data_fp16,
    input  logic                         weight_last,
    input  logic                         weight_commit,

    input  logic                         start_valid,
    output logic                         start_ready,
    input  logic [1:0]                   start_matrix_kind,
    input  logic [LEN_W-1:0]             start_input_length,
    input  logic [LEN_W-1:0]             start_output_length,
    input  logic [META_W-1:0]            start_meta,

    output logic                         output_valid,
    input  logic                         output_ready,
    output logic [1:0]                   output_matrix_kind,
    output logic [DIM_W-1:0]             output_index,
    output logic [31:0]                  output_data_fp32,
    output logic [7:0]                   output_status,
    output logic                         output_invalid,
    output logic [META_W-1:0]            output_meta,
    output logic                         output_last,

    output logic                         done_valid,
    input  logic                         done_ready,
    output logic [7:0]                   done_status,
    output logic                         done_invalid,
    output logic [META_W-1:0]            done_meta,

    output logic [COUNTER_W-1:0]         perf_total_cycles,
    output logic [COUNTER_W-1:0]         perf_pe_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_weight_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_output_stall_cycles
);
    localparam logic [7:0] STATUS_OK = 8'h00;
    localparam logic [7:0] STATUS_INCOMPLETE = 8'hA1;
    localparam logic [7:0] STATUS_RANGE = 8'hA2;

    typedef enum logic [1:0] {
        ST_IDLE,
        ST_START_ROW,
        ST_WAIT_ROW,
        ST_DONE
    } state_e;

    state_e state_q;

    logic input_clear;
    logic [D_MODEL*16-1:0] input_vector_flat;
    logic [D_MODEL-1:0] input_loaded_mask;
    logic input_complete;
    logic input_error;

    logic weight_clear;
    logic [D_MODEL*16-1:0] weight_row_flat;
    logic [D_MODEL-1:0] weight_row_loaded_mask;
    logic [3:0] matrix_complete;
    logic weight_error;
    logic [1:0] read_kind;
    logic [DIM_W-1:0] read_output_index;

    logic [1:0] active_kind_q;
    logic [LEN_W-1:0] active_input_length_q;
    logic [LEN_W-1:0] active_output_length_q;
    logic [DIM_W-1:0] row_index_q;
    logic [META_W-1:0] meta_q;
    logic [7:0] status_q;
    logic invalid_q;

    logic core_command_valid;
    logic core_command_ready;
    logic core_output_valid;
    logic core_output_ready;
    logic [1:0] core_output_matrix_kind;
    logic [DIM_W-1:0] core_output_index;
    logic [31:0] core_output_data_fp32;
    logic [PE_NUM-1:0] core_output_lane_mask;
    logic [7:0] core_output_status;
    logic core_output_invalid;
    logic [META_W-1:0] core_output_meta;
    logic core_output_last;
    logic [COUNTER_W-1:0] core_perf_total_cycles;
    logic [COUNTER_W-1:0] core_perf_tile_cycles;
    logic [COUNTER_W-1:0] core_perf_pe_stall_cycles;
    logic [COUNTER_W-1:0] core_perf_output_stall_cycles;

    logic done_valid_q;
    logic [7:0] done_status_q;
    logic done_invalid_q;
    logic [META_W-1:0] done_meta_q;

    wire start_fire = start_valid && start_ready;
    wire core_command_fire = core_command_valid && core_command_ready;
    wire core_output_fire = core_output_valid && core_output_ready;
    wire done_fire = done_valid && done_ready;
    wire start_kind_complete = int'(start_matrix_kind) < 4 && matrix_complete[int'(start_matrix_kind)];
    wire start_range_legal =
        (start_input_length > LEN_W'(0)) && (start_input_length <= LEN_W'(D_MODEL)) &&
        (start_output_length > LEN_W'(0)) && (start_output_length <= LEN_W'(D_MODEL));
    wire row_last = (LEN_W'(row_index_q) + LEN_W'(1)) == active_output_length_q;

    initial begin
        if (D_MODEL <= 0 || PE_NUM <= 0 || META_W <= 0 || COUNTER_W <= 0) begin
            $fatal(1, "projection_controller parameters must be positive");
        end
    end

    assign input_clear = 1'b0;
    assign weight_clear = 1'b0;
    assign input_ready = (state_q == ST_IDLE);
    assign weight_ready = (state_q == ST_IDLE);
    assign start_ready = (state_q == ST_IDLE) && !done_valid_q;
    assign read_kind = active_kind_q;
    assign read_output_index = row_index_q;
    assign core_command_valid = state_q == ST_START_ROW;
    assign core_output_ready = (state_q == ST_WAIT_ROW) && output_ready;

    assign output_valid = (state_q == ST_WAIT_ROW) && core_output_valid;
    assign output_matrix_kind = core_output_matrix_kind;
    assign output_index = core_output_index;
    assign output_data_fp32 = core_output_data_fp32;
    assign output_status = core_output_status | status_q;
    assign output_invalid = core_output_invalid | invalid_q;
    assign output_meta = core_output_meta;
    assign output_last = core_output_last && row_last;

    assign done_valid = done_valid_q;
    assign done_status = done_status_q;
    assign done_invalid = done_invalid_q;
    assign done_meta = done_meta_q;

    projection_input_buffer #(
        .D_MODEL(D_MODEL)
    ) u_input_buffer (
        .clk              (clk),
        .rst_n            (rst_n),
        .clear            (input_clear),
        .load_valid       (input_valid && input_ready),
        .load_ready       (),
        .load_index       (input_dim),
        .load_data_fp16   (input_data_fp16),
        .load_last        (input_last),
        .load_commit      (input_commit),
        .vector_flat_fp16 (input_vector_flat),
        .loaded_mask      (input_loaded_mask),
        .complete         (input_complete),
        .error_valid      (input_error)
    );

    projection_weight_buffer #(
        .D_MODEL(D_MODEL),
        .MATRIX_KIND_N(4)
    ) u_weight_buffer (
        .clk                  (clk),
        .rst_n                (rst_n),
        .clear                (weight_clear),
        .weight_valid         (weight_valid && weight_ready),
        .weight_ready         (),
        .weight_kind          (weight_kind),
        .weight_output_index  (weight_output_index),
        .weight_input_index   (weight_input_index),
        .weight_data_fp16     (weight_data_fp16),
        .weight_last          (weight_last),
        .weight_commit        (weight_commit),
        .read_kind            (read_kind),
        .read_output_index    (read_output_index),
        .read_row_flat_fp16   (weight_row_flat),
        .read_row_loaded_mask (weight_row_loaded_mask),
        .matrix_complete      (matrix_complete),
        .error_valid          (weight_error)
    );

    shared_gemv_projection_core #(
        .D_MODEL(D_MODEL),
        .PE_NUM(PE_NUM),
        .META_W(META_W),
        .COUNTER_W(COUNTER_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_shared_gemv_projection_core (
        .clk                       (clk),
        .rst_n                     (rst_n),
        .command_valid             (core_command_valid),
        .command_ready             (core_command_ready),
        .command_matrix_kind       (active_kind_q),
        .command_input_length      (active_input_length_q),
        .command_output_index      (row_index_q),
        .command_input_vector_fp16 (input_vector_flat),
        .command_weight_row_fp16   (weight_row_flat),
        .command_meta              (meta_q),
        .command_last              (row_last),
        .output_valid              (core_output_valid),
        .output_ready              (core_output_ready),
        .output_matrix_kind        (core_output_matrix_kind),
        .output_index              (core_output_index),
        .output_data_fp32          (core_output_data_fp32),
        .output_lane_mask          (core_output_lane_mask),
        .output_status             (core_output_status),
        .output_invalid            (core_output_invalid),
        .output_meta               (core_output_meta),
        .output_last               (core_output_last),
        .perf_total_cycles         (core_perf_total_cycles),
        .perf_tile_cycles          (core_perf_tile_cycles),
        .perf_pe_stall_cycles      (core_perf_pe_stall_cycles),
        .perf_output_stall_cycles  (core_perf_output_stall_cycles)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            active_kind_q <= 2'd0;
            active_input_length_q <= '0;
            active_output_length_q <= '0;
            row_index_q <= '0;
            meta_q <= '0;
            status_q <= STATUS_OK;
            invalid_q <= 1'b0;
            done_valid_q <= 1'b0;
            done_status_q <= STATUS_OK;
            done_invalid_q <= 1'b0;
            done_meta_q <= '0;
        end else begin
            if (done_fire) begin
                done_valid_q <= 1'b0;
                done_status_q <= STATUS_OK;
                done_invalid_q <= 1'b0;
                done_meta_q <= '0;
            end

            unique case (state_q)
                ST_IDLE: begin
                    if (start_fire) begin
                        active_kind_q <= start_matrix_kind;
                        active_input_length_q <= start_input_length;
                        active_output_length_q <= start_output_length;
                        row_index_q <= '0;
                        meta_q <= start_meta;
                        status_q <= STATUS_OK;
                        invalid_q <= 1'b0;
                        if (!start_range_legal) begin
                            done_valid_q <= 1'b1;
                            done_status_q <= STATUS_RANGE;
                            done_invalid_q <= 1'b1;
                            done_meta_q <= start_meta;
                            state_q <= ST_DONE;
                        end else if (!input_complete || !start_kind_complete) begin
                            done_valid_q <= 1'b1;
                            done_status_q <= STATUS_INCOMPLETE;
                            done_invalid_q <= 1'b1;
                            done_meta_q <= start_meta;
                            state_q <= ST_DONE;
                        end else begin
                            state_q <= ST_START_ROW;
                        end
                    end
                end

                ST_START_ROW: begin
                    if (core_command_fire) begin
                        state_q <= ST_WAIT_ROW;
                    end
                end

                ST_WAIT_ROW: begin
                    if (core_output_fire) begin
                        status_q <= status_q | core_output_status;
                        invalid_q <= invalid_q | core_output_invalid;
                        if (row_last) begin
                            done_valid_q <= 1'b1;
                            done_status_q <= status_q | core_output_status;
                            done_invalid_q <= invalid_q | core_output_invalid;
                            done_meta_q <= meta_q;
                            state_q <= ST_DONE;
                        end else begin
                            row_index_q <= row_index_q + DIM_W'(1);
                            state_q <= ST_START_ROW;
                        end
                    end
                end

                ST_DONE: begin
                    if (!done_valid_q || done_ready) begin
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
            perf_pe_stall_cycles <= '0;
            perf_weight_stall_cycles <= '0;
            perf_output_stall_cycles <= '0;
        end else begin
            if ((state_q != ST_IDLE) || start_valid) begin
                perf_total_cycles <= perf_total_cycles + COUNTER_W'(1);
            end
            perf_pe_stall_cycles <= core_perf_pe_stall_cycles;
            if ((weight_valid && !weight_ready) || (input_valid && !input_ready)) begin
                perf_weight_stall_cycles <= perf_weight_stall_cycles + COUNTER_W'(1);
            end
            perf_output_stall_cycles <= core_perf_output_stall_cycles;
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n) begin
            assert (!(start_fire && !input_complete))
                else $error("projection_controller no_projection_start_without_complete_input failed");
            assert (!(start_fire && !start_kind_complete))
                else $error("projection_controller weight_matrix_complete_before_start failed");
            assert (!(output_valid && $isunknown({output_matrix_kind, output_index, output_data_fp32,
                                                  output_status, output_invalid, output_meta, output_last})))
                else $error("projection_controller no_unknown_output_when_valid failed");
            if ($past(rst_n) && $past(output_valid && !output_ready)) begin
                assert (output_valid)
                    else $error("projection_controller output valid dropped under backpressure");
                assert ($stable({output_matrix_kind, output_index, output_data_fp32,
                                 output_status, output_invalid, output_meta, output_last}))
                    else $error("projection_controller output stable until ready failed");
            end
        end
    end
`endif
endmodule

`default_nettype wire
