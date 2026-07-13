`default_nettype none

module head_concat_quantizer #(
    parameter int N_HEAD = 2,
    parameter int D_HEAD = 8,
    parameter int PE_NUM = 8,
    parameter int META_W = 16,
    parameter int COUNTER_W = 64,
    parameter bit ASSERT_ON_INVALID = 1'b1,
    localparam int D_MODEL = N_HEAD * D_HEAD,
    localparam int HEAD_W = (N_HEAD <= 1) ? 1 : $clog2(N_HEAD),
    localparam int DIM_W = (D_HEAD <= 1) ? 1 : $clog2(D_HEAD),
    localparam int MODEL_W = (D_MODEL <= 1) ? 1 : $clog2(D_MODEL),
    localparam int LANE_W = (PE_NUM <= 1) ? 1 : $clog2(PE_NUM)
) (
    input  logic                         clk,
    input  logic                         rst_n,

    input  logic                         clear,

    input  logic                         input_valid,
    output logic                         input_ready,
    input  logic [HEAD_W-1:0]            input_head,
    input  logic [DIM_W-1:0]             input_base_dim,
    input  logic [PE_NUM*32-1:0]         input_vector_fp32,
    input  logic [PE_NUM-1:0]            input_lane_mask,
    input  logic                         input_last_tile,
    input  logic                         input_last_head,
    input  logic                         input_last_token,
    input  logic [7:0]                   input_status,
    input  logic                         input_invalid,
    input  logic [META_W-1:0]            input_meta,

    output logic                         write_valid,
    input  logic                         write_ready,
    output logic [MODEL_W-1:0]           write_index,
    output logic [15:0]                  write_data_fp16,

    output logic                         concat_complete,
    output logic [7:0]                   concat_status,
    output logic                         concat_invalid,
    output logic [META_W-1:0]            concat_meta,
    output logic                         busy,
    output logic [COUNTER_W-1:0]         perf_concat_quantization_cycles,
    output logic [COUNTER_W-1:0]         perf_output_stall_cycles
);
    localparam logic [7:0] STATUS_OK = 8'h00;
    localparam logic [7:0] STATUS_ORDER = 8'hC1;
    localparam logic [7:0] STATUS_LANE = 8'hC2;
    localparam int META_OUT_W = MODEL_W;

    typedef enum logic [1:0] {
        ST_IDLE,
        ST_SEND_LANE,
        ST_WAIT_QUANT
    } state_e;

    state_e state_q;

    logic [HEAD_W-1:0] expected_head_q;
    logic [DIM_W-1:0] expected_base_q;
    logic [HEAD_W-1:0] tile_head_q;
    logic [DIM_W-1:0] tile_base_q;
    logic [PE_NUM*32-1:0] tile_vector_q;
    logic [PE_NUM-1:0] tile_mask_q;
    logic tile_last_tile_q;
    logic tile_last_head_q;
    logic tile_last_token_q;
    logic [LANE_W-1:0] lane_q;
    logic [7:0] status_q;
    logic invalid_q;
    logic [META_W-1:0] meta_q;
    logic complete_q;
    logic seen_first_q;

    logic quant_in_valid;
    logic quant_in_ready;
    logic quant_out_valid;
    logic quant_out_ready;
    logic [15:0] quant_out_data;
    logic quant_out_invalid;
    logic quant_out_overflow;
    logic quant_out_underflow_or_ftz;
    logic quant_out_inexact;
    logic [META_OUT_W-1:0] quant_out_meta;
    logic quant_out_last;

    wire input_fire = input_valid && input_ready;
    wire quant_in_fire = quant_in_valid && quant_in_ready;
    wire quant_out_fire = quant_out_valid && quant_out_ready;
    wire [DIM_W:0] tile_end_dim_ext = {1'b0, input_base_dim} + (DIM_W+1)'(PE_NUM);
    wire input_last_tile_expected = tile_end_dim_ext >= (DIM_W+1)'(D_HEAD);
    wire input_last_head_expected =
        (input_head == HEAD_W'(N_HEAD - 1)) && input_last_tile_expected;
    wire order_legal =
        (input_head == expected_head_q) &&
        (input_base_dim == expected_base_q) &&
        (input_last_tile == input_last_tile_expected) &&
        (input_last_head == input_last_head_expected) &&
        (input_last_token == input_last_head_expected);
    wire lane_in_range = (int'(tile_base_q) + int'(lane_q)) < D_HEAD;
    wire lane_active = tile_mask_q[int'(lane_q)] && lane_in_range;
    wire [MODEL_W-1:0] lane_concat_index =
        MODEL_W'(int'(tile_head_q) * D_HEAD + int'(tile_base_q) + int'(lane_q));
    wire at_last_lane = int'(lane_q) == (PE_NUM - 1);

    initial begin
        if (N_HEAD <= 0 || D_HEAD <= 0 || PE_NUM <= 0 || META_W <= 0 || COUNTER_W <= 0) begin
            $fatal(1, "head_concat_quantizer parameters must be positive");
        end
        if (D_MODEL != N_HEAD * D_HEAD) begin
            $fatal(1, "head_concat_quantizer d_model_equals_n_head_times_d_head failed");
        end
    end

    assign input_ready = (state_q == ST_IDLE) && !complete_q;
    assign busy = (state_q != ST_IDLE);
    assign concat_complete = complete_q;
    assign concat_status = status_q;
    assign concat_invalid = invalid_q;
    assign concat_meta = meta_q;

    assign quant_in_valid = (state_q == ST_SEND_LANE) && lane_active;
    assign quant_out_ready = write_ready;
    assign write_valid = quant_out_valid;
    assign write_index = quant_out_meta;
    assign write_data_fp16 = quant_out_data;

    fp32_to_fp16 #(
        .META_W(META_OUT_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_concat_quantizer (
        .clk                  (clk),
        .rst_n                (rst_n),
        .in_valid             (quant_in_valid),
        .in_ready             (quant_in_ready),
        .in_data              (tile_vector_q[int'(lane_q)*32 +: 32]),
        .in_meta              (lane_concat_index),
        .in_last              (tile_last_tile_q && tile_last_head_q && tile_last_token_q && at_last_lane),
        .out_valid            (quant_out_valid),
        .out_ready            (quant_out_ready),
        .out_data             (quant_out_data),
        .out_invalid          (quant_out_invalid),
        .out_overflow         (quant_out_overflow),
        .out_underflow_or_ftz (quant_out_underflow_or_ftz),
        .out_inexact          (quant_out_inexact),
        .out_meta             (quant_out_meta),
        .out_last             (quant_out_last)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            expected_head_q <= '0;
            expected_base_q <= '0;
            tile_head_q <= '0;
            tile_base_q <= '0;
            tile_vector_q <= '0;
            tile_mask_q <= '0;
            tile_last_tile_q <= 1'b0;
            tile_last_head_q <= 1'b0;
            tile_last_token_q <= 1'b0;
            lane_q <= '0;
            status_q <= STATUS_OK;
            invalid_q <= 1'b0;
            meta_q <= '0;
            complete_q <= 1'b0;
            seen_first_q <= 1'b0;
        end else begin
            if (clear) begin
                state_q <= ST_IDLE;
                expected_head_q <= '0;
                expected_base_q <= '0;
                lane_q <= '0;
                status_q <= STATUS_OK;
                invalid_q <= 1'b0;
                meta_q <= '0;
                complete_q <= 1'b0;
                seen_first_q <= 1'b0;
            end else begin
                if (quant_out_fire) begin
                    status_q <= status_q | {4'd0, quant_out_overflow, quant_out_underflow_or_ftz,
                                            quant_out_inexact, 1'b0};
                    invalid_q <= invalid_q | quant_out_invalid;
                end

                unique case (state_q)
                    ST_IDLE: begin
                        if (input_fire) begin
                            tile_head_q <= input_head;
                            tile_base_q <= input_base_dim;
                            tile_vector_q <= input_vector_fp32;
                            tile_mask_q <= input_lane_mask;
                            tile_last_tile_q <= input_last_tile;
                            tile_last_head_q <= input_last_head;
                            tile_last_token_q <= input_last_token;
                            lane_q <= '0;
                            if (!seen_first_q) begin
                                meta_q <= input_meta;
                                seen_first_q <= 1'b1;
                            end
                            status_q <= status_q | input_status;
                            invalid_q <= invalid_q | input_invalid;
                            if (!order_legal) begin
                                status_q <= status_q | STATUS_ORDER;
                                invalid_q <= 1'b1;
                            end
                            for (int lane = 0; lane < PE_NUM; lane++) begin
                                if ((int'(input_base_dim) + lane) >= D_HEAD && input_lane_mask[lane]) begin
                                    status_q <= status_q | STATUS_LANE;
                                    invalid_q <= 1'b1;
                                end
                            end
                            state_q <= ST_SEND_LANE;
                        end
                    end

                    ST_SEND_LANE: begin
                        if (!lane_active) begin
                            if (at_last_lane) begin
                                if (tile_last_tile_q && tile_last_head_q && tile_last_token_q) begin
                                    complete_q <= 1'b1;
                                end else if (tile_last_tile_q) begin
                                    expected_head_q <= expected_head_q + HEAD_W'(1);
                                    expected_base_q <= '0;
                                end else begin
                                    expected_base_q <= expected_base_q + DIM_W'(PE_NUM);
                                end
                                state_q <= ST_IDLE;
                            end else begin
                                lane_q <= lane_q + LANE_W'(1);
                            end
                        end else if (quant_in_fire) begin
                            state_q <= ST_WAIT_QUANT;
                        end
                    end

                    ST_WAIT_QUANT: begin
                        if (quant_out_fire) begin
                            if (at_last_lane) begin
                                if (tile_last_tile_q && tile_last_head_q && tile_last_token_q) begin
                                    complete_q <= 1'b1;
                                end else if (tile_last_tile_q) begin
                                    expected_head_q <= expected_head_q + HEAD_W'(1);
                                    expected_base_q <= '0;
                                end else begin
                                    expected_base_q <= expected_base_q + DIM_W'(PE_NUM);
                                end
                                state_q <= ST_IDLE;
                            end else begin
                                lane_q <= lane_q + LANE_W'(1);
                                state_q <= ST_SEND_LANE;
                            end
                        end
                    end

                    default: state_q <= ST_IDLE;
                endcase
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            perf_concat_quantization_cycles <= '0;
            perf_output_stall_cycles <= '0;
        end else begin
            if (busy || input_valid || quant_out_valid) begin
                perf_concat_quantization_cycles <= perf_concat_quantization_cycles + COUNTER_W'(1);
            end
            if (quant_out_valid && !write_ready) begin
                perf_output_stall_cycles <= perf_output_stall_cycles + COUNTER_W'(1);
            end
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n) begin
            assert (!(input_fire && !order_legal))
                else $error("head_concat_quantizer no_concat_before_head_output/order failed");
            assert (!(quant_in_fire && !lane_in_range))
                else $error("head_concat_quantizer concat_index_in_range failed");
            assert (!(quant_in_fire && !tile_mask_q[int'(lane_q)]))
                else $error("head_concat_quantizer concat_write_only_for_active_lane failed");
            if ($past(rst_n) && $past(input_valid && !input_ready)) begin
                assert (input_valid)
                    else $error("head_concat_quantizer input valid dropped under backpressure");
                assert ($stable({input_head, input_base_dim, input_vector_fp32, input_lane_mask,
                                 input_last_tile, input_last_head, input_last_token, input_status,
                                 input_invalid, input_meta}))
                    else $error("head_concat_quantizer input payload changed under backpressure");
            end
        end
    end
`endif
endmodule

`default_nettype wire
