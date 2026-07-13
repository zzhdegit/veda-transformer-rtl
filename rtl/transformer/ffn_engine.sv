`default_nettype none

module ffn_engine #(
    parameter int D_MODEL = 16,
    parameter int PE_NUM = 8,
    parameter int META_W = 16,
    parameter int COUNTER_W = 64,
    parameter bit ASSERT_ON_INVALID = 1'b1,
    localparam int D_FFN = 4 * D_MODEL,
    localparam int MODEL_W = (D_MODEL <= 1) ? 1 : $clog2(D_MODEL),
    localparam int FFN_W = (D_FFN <= 1) ? 1 : $clog2(D_FFN),
    localparam int LEN_W = (D_FFN <= 1) ? 1 : $clog2(D_FFN + 1),
    localparam int LANE_COUNT_W = $clog2(PE_NUM + 1)
) (
    input  logic                         clk,
    input  logic                         rst_n,

    input  logic                         clear,

    input  logic                         weight_valid,
    output logic                         weight_ready,
    input  logic                         weight_kind,
    input  logic [FFN_W-1:0]             weight_output_index,
    input  logic [FFN_W-1:0]             weight_input_index,
    input  logic [15:0]                  weight_data_fp16,
    input  logic                         weight_commit,

    input  logic                         input_valid,
    output logic                         input_ready,
    input  logic [MODEL_W-1:0]           input_dim,
    input  logic [15:0]                  input_data_fp16,
    input  logic                         input_last,
    input  logic [META_W-1:0]            input_meta,
    input  logic                         input_commit,

    input  logic                         start_valid,
    output logic                         start_ready,
    input  logic [META_W-1:0]            start_meta,

    output logic                         output_valid,
    input  logic                         output_ready,
    output logic [MODEL_W-1:0]           output_dim,
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

    output logic [COUNTER_W-1:0]         perf_ffn1_cycles,
    output logic [COUNTER_W-1:0]         perf_relu_cycles,
    output logic [COUNTER_W-1:0]         perf_activation_quantization_cycles,
    output logic [COUNTER_W-1:0]         perf_ffn2_cycles,
    output logic [COUNTER_W-1:0]         perf_pe_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_output_stall_cycles
);
    localparam logic [1:0] MODE_GEMV = 2'd0;
    localparam logic [7:0] STATUS_OK = 8'h00;
    localparam logic [7:0] STATUS_INCOMPLETE = 8'hD1;
    localparam logic [7:0] STATUS_RANGE = 8'hD2;
    localparam logic [31:0] FP32_ZERO = 32'h0000_0000;

    typedef enum logic [3:0] {
        ST_IDLE,
        ST_FFN1_SEND_TILE,
        ST_FFN1_WAIT_OUTPUT,
        ST_RELU_QUANT_SEND,
        ST_RELU_QUANT_WAIT,
        ST_FFN2_SEND_TILE,
        ST_FFN2_WAIT_OUTPUT,
        ST_DONE
    } state_e;

    state_e state_q = ST_IDLE;

    logic [15:0] input_mem [0:D_MODEL-1];
    logic [15:0] w1_mem [0:D_FFN-1][0:D_MODEL-1];
    logic [15:0] w2_mem [0:D_MODEL-1][0:D_FFN-1];
    logic [15:0] activation_mem [0:D_FFN-1];
    logic [D_MODEL-1:0] input_loaded_mask_q;
    logic input_complete_q;
    logic w1_complete_q;
    logic w2_complete_q;
    logic load_error_q;

    logic [FFN_W-1:0] row_index_q;
    logic [LEN_W-1:0] tile_base_q;
    logic [META_W-1:0] meta_q;
    logic [7:0] status_q;
    logic invalid_q;
    logic [31:0] ffn1_q;

    logic done_valid_q;
    logic [7:0] done_status_q;
    logic done_invalid_q;
    logic [META_W-1:0] done_meta_q;

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

    logic quant_in_valid;
    logic quant_in_ready;
    logic quant_out_valid;
    logic quant_out_ready;
    logic [15:0] quant_out_data;
    logic quant_out_invalid;
    logic quant_out_overflow;
    logic quant_out_underflow_or_ftz;
    logic quant_out_inexact;

    logic relu_invalid;
    logic [31:0] relu_fp32;

    wire weight_fire = weight_valid && weight_ready;
    wire input_fire = input_valid && input_ready;
    wire start_fire = start_valid && start_ready;
    wire done_fire = done_valid && done_ready;
    wire pe_in_fire = pe_in_valid && pe_in_ready;
    wire pe_out_fire = pe_out_valid && pe_out_ready;
    wire quant_out_fire = quant_out_valid && quant_out_ready;
    wire in_ffn1 = state_q == ST_FFN1_SEND_TILE || state_q == ST_FFN1_WAIT_OUTPUT;
    wire [LEN_W-1:0] active_input_length = in_ffn1 ? LEN_W'(D_MODEL) : LEN_W'(D_FFN);
    wire [LEN_W:0] tile_remaining = {1'b0, active_input_length} - {1'b0, tile_base_q};
    wire [LEN_W:0] tile_width_ext =
        (tile_remaining > (LEN_W+1)'(PE_NUM)) ? (LEN_W+1)'(PE_NUM) : tile_remaining;
    wire ffn1_last_row = row_index_q == FFN_W'(D_FFN - 1);
    wire ffn2_last_row = row_index_q == FFN_W'(D_MODEL - 1);
    wire input_range_legal = int'(input_dim) < D_MODEL;
    wire w1_range_legal = !weight_kind &&
                          int'(weight_output_index) < D_FFN &&
                          int'(weight_input_index) < D_MODEL;
    wire w2_range_legal = weight_kind &&
                          int'(weight_output_index) < D_MODEL &&
                          int'(weight_input_index) < D_FFN;

    initial begin
        if (D_MODEL <= 0 || PE_NUM <= 0 || META_W <= 0 || COUNTER_W <= 0) begin
            $fatal(1, "ffn_engine parameters must be positive");
        end
        if ((PE_NUM & (PE_NUM - 1)) != 0) begin
            $fatal(1, "ffn_engine PE_NUM must be power of two");
        end
    end

    assign weight_ready = state_q == ST_IDLE;
    assign input_ready = state_q == ST_IDLE;
    assign start_ready = (state_q == ST_IDLE) && !done_valid_q;

    assign pe_in_valid = state_q == ST_FFN1_SEND_TILE || state_q == ST_FFN2_SEND_TILE;
    assign pe_in_tile_first = tile_base_q == LEN_W'(0);
    assign pe_in_tile_last = (tile_base_q + LEN_W'(PE_NUM)) >= active_input_length;
    assign pe_in_active_lanes = (state_q == ST_FFN1_SEND_TILE || state_q == ST_FFN2_SEND_TILE) ?
                                LANE_COUNT_W'(tile_width_ext) : '0;
    assign pe_out_ready = (state_q == ST_FFN1_WAIT_OUTPUT) ||
                          ((state_q == ST_FFN2_WAIT_OUTPUT) && output_ready);

    assign relu_invalid = ffn1_q[30:23] == 8'hFF;
    assign relu_fp32 = relu_invalid ? FP32_ZERO :
                       (ffn1_q[31] ? FP32_ZERO :
                        (((ffn1_q & 32'h7FFF_FFFF) == 32'd0) ? FP32_ZERO : ffn1_q));
    assign quant_in_valid = state_q == ST_RELU_QUANT_SEND;
    assign quant_out_ready = state_q == ST_RELU_QUANT_WAIT;

    assign output_valid = (state_q == ST_FFN2_WAIT_OUTPUT) && pe_out_valid;
    assign output_dim = MODEL_W'(row_index_q);
    assign output_data_fp32 = pe_out_scalar;
    assign output_status = status_q | pe_out_status;
    assign output_invalid = invalid_q | pe_out_invalid;
    assign output_meta = meta_q;
    assign output_last = ffn2_last_row;

    assign done_valid = done_valid_q;
    assign done_status = done_status_q;
    assign done_invalid = done_invalid_q;
    assign done_meta = done_meta_q;

    always_comb begin
        pe_in_vector_a = '0;
        pe_in_vector_b = '0;
        if (state_q == ST_FFN1_SEND_TILE) begin
            for (int lane = 0; lane < PE_NUM; lane++) begin
                if ((int'(tile_base_q) + lane) < D_MODEL) begin
                    pe_in_vector_a[lane*16 +: 16] = input_mem[int'(tile_base_q) + lane];
                    pe_in_vector_b[lane*16 +: 16] = w1_mem[int'(row_index_q)][int'(tile_base_q) + lane];
                end
            end
        end else if (state_q == ST_FFN2_SEND_TILE) begin
            for (int lane = 0; lane < PE_NUM; lane++) begin
                if ((int'(tile_base_q) + lane) < D_FFN) begin
                    pe_in_vector_a[lane*16 +: 16] = activation_mem[int'(tile_base_q) + lane];
                    pe_in_vector_b[lane*16 +: 16] = w2_mem[int'(row_index_q)][int'(tile_base_q) + lane];
                end
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
        .in_last                     (in_ffn1 ? ffn1_last_row : ffn2_last_row),
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

    fp32_to_fp16 #(
        .META_W(META_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_activation_quant (
        .clk                  (clk),
        .rst_n                (rst_n),
        .in_valid             (quant_in_valid),
        .in_ready             (quant_in_ready),
        .in_data              (relu_fp32),
        .in_meta              (meta_q),
        .in_last              (ffn1_last_row),
        .out_valid            (quant_out_valid),
        .out_ready            (quant_out_ready),
        .out_data             (quant_out_data),
        .out_invalid          (quant_out_invalid),
        .out_overflow         (quant_out_overflow),
        .out_underflow_or_ftz (quant_out_underflow_or_ftz),
        .out_inexact          (quant_out_inexact),
        .out_meta             (),
        .out_last             ()
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            input_loaded_mask_q <= '0;
            input_complete_q <= 1'b0;
            w1_complete_q <= 1'b0;
            w2_complete_q <= 1'b0;
            load_error_q <= 1'b0;
            row_index_q <= '0;
            tile_base_q <= '0;
            meta_q <= '0;
            status_q <= STATUS_OK;
            invalid_q <= 1'b0;
            ffn1_q <= FP32_ZERO;
            done_valid_q <= 1'b0;
            done_status_q <= STATUS_OK;
            done_invalid_q <= 1'b0;
            done_meta_q <= '0;
            perf_ffn1_cycles <= '0;
            perf_relu_cycles <= '0;
            perf_activation_quantization_cycles <= '0;
            perf_ffn2_cycles <= '0;
            perf_pe_stall_cycles <= '0;
            perf_output_stall_cycles <= '0;
            for (int dim = 0; dim < D_MODEL; dim++) begin
                input_mem[dim] <= 16'd0;
                for (int col = 0; col < D_FFN; col++) begin
                    w2_mem[dim][col] <= 16'd0;
                end
            end
            for (int row = 0; row < D_FFN; row++) begin
                activation_mem[row] <= 16'd0;
                for (int col = 0; col < D_MODEL; col++) begin
                    w1_mem[row][col] <= 16'd0;
                end
            end
        end else begin
            if (clear) begin
                state_q <= ST_IDLE;
                input_loaded_mask_q <= '0;
                input_complete_q <= 1'b0;
                load_error_q <= 1'b0;
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
                    state_q <= ST_IDLE;
                end

                if (weight_fire) begin
                    if (w1_range_legal) begin
                        w1_mem[int'(weight_output_index)][int'(weight_input_index)] <= weight_data_fp16;
                    end else if (w2_range_legal) begin
                        w2_mem[int'(weight_output_index)][int'(weight_input_index)] <= weight_data_fp16;
                    end else begin
                        load_error_q <= 1'b1;
                    end
                    if (weight_commit) begin
                        if (!weight_kind) begin
                            w1_complete_q <= 1'b1;
                        end else begin
                            w2_complete_q <= 1'b1;
                        end
                    end
                end

                if (input_fire) begin
                    meta_q <= input_meta;
                    if (input_range_legal) begin
                        input_mem[int'(input_dim)] <= input_data_fp16;
                        input_loaded_mask_q[int'(input_dim)] <= 1'b1;
                    end else begin
                        load_error_q <= 1'b1;
                    end
                    if (input_commit) begin
                        input_complete_q <= 1'b1;
                    end
                    if (input_last && input_dim != MODEL_W'(D_MODEL - 1)) begin
                        load_error_q <= 1'b1;
                    end
                end

                if (!done_fire) begin
                    unique case (state_q)
                        ST_IDLE: begin
                            if (start_fire) begin
                                meta_q <= start_meta;
                                status_q <= STATUS_OK;
                                invalid_q <= 1'b0;
                                row_index_q <= '0;
                                tile_base_q <= '0;
                                if (!input_complete_q || !w1_complete_q || !w2_complete_q ||
                                    input_loaded_mask_q != {D_MODEL{1'b1}}) begin
                                    done_valid_q <= 1'b1;
                                    done_status_q <= STATUS_INCOMPLETE;
                                    done_invalid_q <= 1'b1;
                                    done_meta_q <= start_meta;
                                    state_q <= ST_DONE;
                                end else if (load_error_q) begin
                                    done_valid_q <= 1'b1;
                                    done_status_q <= STATUS_RANGE;
                                    done_invalid_q <= 1'b1;
                                    done_meta_q <= start_meta;
                                    state_q <= ST_DONE;
                                end else begin
                                    state_q <= ST_FFN1_SEND_TILE;
                                end
                            end
                        end

                        ST_FFN1_SEND_TILE: begin
                            if (pe_in_fire) begin
                                if (pe_in_tile_last) begin
                                    state_q <= ST_FFN1_WAIT_OUTPUT;
                                end else begin
                                    tile_base_q <= tile_base_q + LEN_W'(PE_NUM);
                                end
                            end
                        end

                        ST_FFN1_WAIT_OUTPUT: begin
                            if (pe_out_fire) begin
                                ffn1_q <= pe_out_scalar;
                                status_q <= status_q | pe_out_status;
                                invalid_q <= invalid_q | pe_out_invalid;
                                state_q <= ST_RELU_QUANT_SEND;
                            end
                        end

                        ST_RELU_QUANT_SEND: begin
                            if (quant_in_ready) begin
                                invalid_q <= invalid_q | relu_invalid;
                                state_q <= ST_RELU_QUANT_WAIT;
                            end
                        end

                        ST_RELU_QUANT_WAIT: begin
                            if (quant_out_fire) begin
                                activation_mem[int'(row_index_q)] <= quant_out_data;
                                invalid_q <= invalid_q | quant_out_invalid;
                                if (ffn1_last_row) begin
                                    row_index_q <= '0;
                                    tile_base_q <= '0;
                                    state_q <= ST_FFN2_SEND_TILE;
                                end else begin
                                    row_index_q <= row_index_q + FFN_W'(1);
                                    tile_base_q <= '0;
                                    state_q <= ST_FFN1_SEND_TILE;
                                end
                            end
                        end

                        ST_FFN2_SEND_TILE: begin
                            if (pe_in_fire) begin
                                if (pe_in_tile_last) begin
                                    state_q <= ST_FFN2_WAIT_OUTPUT;
                                end else begin
                                    tile_base_q <= tile_base_q + LEN_W'(PE_NUM);
                                end
                            end
                        end

                        ST_FFN2_WAIT_OUTPUT: begin
                            if (output_valid && !output_ready) begin
                                perf_output_stall_cycles <= perf_output_stall_cycles + COUNTER_W'(1);
                            end
                            if (pe_out_fire) begin
                                status_q <= status_q | pe_out_status;
                                invalid_q <= invalid_q | pe_out_invalid;
                                if (ffn2_last_row) begin
                                    done_valid_q <= 1'b1;
                                    done_status_q <= status_q | pe_out_status;
                                    done_invalid_q <= invalid_q | pe_out_invalid;
                                    done_meta_q <= meta_q;
                                    state_q <= ST_DONE;
                                end else begin
                                    row_index_q <= row_index_q + FFN_W'(1);
                                    tile_base_q <= '0;
                                    state_q <= ST_FFN2_SEND_TILE;
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

                if (state_q == ST_FFN1_SEND_TILE || state_q == ST_FFN1_WAIT_OUTPUT) begin
                    perf_ffn1_cycles <= perf_ffn1_cycles + COUNTER_W'(1);
                end
                if (state_q == ST_RELU_QUANT_SEND) begin
                    perf_relu_cycles <= perf_relu_cycles + COUNTER_W'(1);
                end
                if (state_q == ST_RELU_QUANT_SEND || state_q == ST_RELU_QUANT_WAIT) begin
                    perf_activation_quantization_cycles <= perf_activation_quantization_cycles + COUNTER_W'(1);
                end
                if (state_q == ST_FFN2_SEND_TILE || state_q == ST_FFN2_WAIT_OUTPUT) begin
                    perf_ffn2_cycles <= perf_ffn2_cycles + COUNTER_W'(1);
                end
                if (pe_in_valid && !pe_in_ready) begin
                    perf_pe_stall_cycles <= perf_pe_stall_cycles + COUNTER_W'(1);
                end
            end
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n) begin
            if (output_valid) begin
                assert (!$isunknown({output_dim, output_data_fp32, output_status,
                                     output_invalid, output_meta, output_last}))
                    else $error("ffn_engine no_unknown_output_when_valid failed");
            end
            if ($past(rst_n) && $past(output_valid && !output_ready)) begin
                assert (output_valid)
                    else $error("ffn_engine output valid dropped under backpressure");
                assert ($stable({output_dim, output_data_fp32, output_status,
                                 output_invalid, output_meta, output_last}))
                    else $error("ffn_engine output stable until ready failed");
            end
        end
    end
`endif
endmodule

`default_nettype wire
