`default_nettype none

module reconfigurable_pe_core #(
    parameter int PE_NUM = 8,
    parameter int META_W = 16,
    parameter int COUNTER_W = 64,
    parameter bit ASSERT_ON_INVALID = 1'b1,
    localparam int LANE_COUNT_W = $clog2(PE_NUM + 1)
) (
    input  logic                         clk,
    input  logic                         rst_n,

    input  logic                         in_valid,
    output logic                         in_ready,
    input  logic [1:0]                   in_mode,          // 0: GEMV/inner, 1: QK inner, 2: SV outer
    input  logic                         in_clear,
    input  logic                         in_tile_first,
    input  logic                         in_tile_last,
    input  logic                         in_use_explicit_mask,
    input  logic [LANE_COUNT_W-1:0]      in_active_lanes,
    input  logic [PE_NUM-1:0]            in_lane_mask,
    input  logic [31:0]                  in_scalar_fp32,
    input  logic [PE_NUM*16-1:0]         in_vector_a_fp16,
    input  logic [PE_NUM*16-1:0]         in_vector_b_fp16,
    input  logic [META_W-1:0]            in_meta,
    input  logic                         in_last,

    output logic                         out_valid,
    input  logic                         out_ready,
    output logic [1:0]                   out_mode,
    output logic [31:0]                  out_scalar_fp32,
    output logic [PE_NUM*32-1:0]         out_vector_fp32,
    output logic [PE_NUM-1:0]            out_lane_mask,
    output logic [7:0]                   out_status,
    output logic                         out_invalid,
    output logic [META_W-1:0]            out_meta,
    output logic                         out_last,

    output logic [COUNTER_W-1:0]         perf_total_cycles,
    output logic [COUNTER_W-1:0]         perf_busy_cycles,
    output logic [COUNTER_W-1:0]         perf_active_lane_cycles,
    output logic [COUNTER_W-1:0]         perf_available_lane_cycles,
    output logic [COUNTER_W-1:0]         perf_input_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_output_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_mode_switch_cycles,
    output logic [COUNTER_W-1:0]         perf_tile_count,
    output logic [COUNTER_W-1:0]         perf_operation_count,
    output logic [COUNTER_W-1:0]         perf_invalid_count
);
    localparam logic [1:0] MODE_GEMV = 2'd0;
    localparam logic [1:0] MODE_QK_INNER = 2'd1;
    localparam logic [1:0] MODE_SV_OUTER = 2'd2;
    localparam logic PE_MODE_PRODUCT = 1'b0;
    localparam logic PE_MODE_FMA = 1'b1;

    typedef enum logic [3:0] {
        ST_IDLE,
        ST_CONV_START,
        ST_CONV_WAIT,
        ST_LANE_WAIT,
        ST_REDUCE_WAIT,
        ST_TILE_ADD_WAIT,
        ST_OUTPUT
    } state_e;

    state_e state_q;

    logic [1:0] mode_q;
    logic tile_first_q;
    logic tile_last_q;
    logic [PE_NUM-1:0] lane_mask_q;
    logic [31:0] scalar_q;
    logic [PE_NUM*16-1:0] vector_a_q;
    logic [PE_NUM*16-1:0] vector_b_q;
    logic [META_W-1:0] meta_q;
    logic last_q;
    logic [31:0] inner_acc_q;
    logic have_inner_partial_q;
    logic have_outer_partial_q;
    logic [1:0] last_mode_q;
    logic last_mode_valid_q;

    logic [7:0] lane_status_or_q;
    logic lane_invalid_or_q;
    logic [7:0] reduce_status_or_q;
    logic reduce_invalid_or_q;

    logic out_valid_q;
    logic [1:0] out_mode_q;
    logic [31:0] out_scalar_q;
    logic [PE_NUM*32-1:0] out_vector_q;
    logic [PE_NUM-1:0] out_lane_mask_q;
    logic [7:0] out_status_q;
    logic out_invalid_q;
    logic [META_W-1:0] out_meta_q;
    logic out_last_q;

    logic [PE_NUM-1:0] effective_lane_mask;
    logic [LANE_COUNT_W-1:0] active_lane_count;

    logic conv_in_valid;
    logic conv_all_in_ready;
    logic conv_all_out_valid;
    logic conv_out_ready;
    logic [PE_NUM-1:0] conv_a_in_ready;
    logic [PE_NUM-1:0] conv_b_in_ready;
    logic [PE_NUM-1:0] conv_a_out_valid;
    logic [PE_NUM-1:0] conv_b_out_valid;
    logic [PE_NUM-1:0] conv_a_invalid;
    logic [PE_NUM-1:0] conv_b_invalid;
    logic [31:0] conv_a_data [0:PE_NUM-1];
    logic [31:0] conv_b_data [0:PE_NUM-1];

    logic lane_in_valid;
    logic lane_all_in_ready;
    logic lane_all_out_valid;
    logic lane_out_ready;
    logic [PE_NUM-1:0] lane_in_ready;
    logic [PE_NUM-1:0] lane_out_valid;
    logic [PE_NUM-1:0] lane_out_active;
    logic [31:0] lane_result [0:PE_NUM-1];
    logic [7:0] lane_status [0:PE_NUM-1];
    logic [PE_NUM-1:0] lane_invalid;
    logic [PE_NUM*32-1:0] lane_result_packed;

    logic reduce_in_valid;
    logic reduce_in_ready;
    logic reduce_out_valid;
    logic reduce_out_ready;
    logic [31:0] reduce_sum;
    logic [7:0] reduce_status;
    logic reduce_invalid;
    logic reduce_busy;

    logic tile_add_in_valid;
    logic tile_add_in_ready;
    logic tile_add_out_valid;
    logic tile_add_out_ready;
    logic [31:0] tile_add_result;
    logic [7:0] tile_add_status;
    logic tile_add_invalid;

    logic bank_clear_valid;
    logic [PE_NUM-1:0] bank_clear_mask;
    logic bank_update_valid;
    logic [PE_NUM-1:0] bank_update_mask;
    logic [PE_NUM*32-1:0] bank_update_values;
    logic [PE_NUM*32-1:0] bank_read_values;

    logic input_fire;
    logic output_fire;
    logic conv_input_fire;
    logic lane_input_fire;
    logic lane_output_fire;
    logic reduce_input_fire;
    logic reduce_output_fire;
    logic tile_add_input_fire;
    logic tile_add_output_fire;
    logic mode_switch_fire;
    logic final_invalid_fire;
    logic [7:0] lane_status_or_comb;
    logic lane_invalid_or_comb;
    logic [PE_NUM*32-1:0] outer_next_vector_comb;

    initial begin
        if (PE_NUM <= 0) begin
            $fatal(1, "reconfigurable_pe_core PE_NUM must be positive");
        end
        if ((PE_NUM & (PE_NUM - 1)) != 0) begin
            $fatal(1, "reconfigurable_pe_core PE_NUM must be a power of two");
        end
        if (META_W <= 0) begin
            $fatal(1, "reconfigurable_pe_core META_W must be positive");
        end
    end

    lane_mask_generator #(
        .PE_NUM(PE_NUM)
    ) u_mask_generator (
        .use_explicit_mask   (in_use_explicit_mask),
        .explicit_lane_mask  (in_lane_mask),
        .active_lanes        (in_active_lanes),
        .lane_mask           (effective_lane_mask)
    );

    function automatic logic [LANE_COUNT_W-1:0] popcount_mask(input logic [PE_NUM-1:0] mask);
        logic [LANE_COUNT_W-1:0] count;
        begin
            count = '0;
            for (int lane = 0; lane < PE_NUM; lane++) begin
                count = count + LANE_COUNT_W'(mask[lane]);
            end
            popcount_mask = count;
        end
    endfunction

    assign active_lane_count = popcount_mask(lane_mask_q);
    assign in_ready = (state_q == ST_IDLE) && !out_valid_q;
    assign input_fire = in_valid && in_ready;
    assign output_fire = out_valid && out_ready;

    assign out_valid = out_valid_q;
    assign out_mode = out_mode_q;
    assign out_scalar_fp32 = out_scalar_q;
    assign out_vector_fp32 = out_vector_q;
    assign out_lane_mask = out_lane_mask_q;
    assign out_status = out_status_q;
    assign out_invalid = out_invalid_q;
    assign out_meta = out_meta_q;
    assign out_last = out_last_q;

    assign conv_in_valid = (state_q == ST_CONV_START);
    assign conv_all_in_ready = (&conv_a_in_ready) && (&conv_b_in_ready);
    assign conv_input_fire = conv_in_valid && conv_all_in_ready;
    assign conv_all_out_valid = (&conv_a_out_valid) && (&conv_b_out_valid);
    assign lane_in_valid = (state_q == ST_CONV_WAIT) && conv_all_out_valid;
    assign lane_all_in_ready = &lane_in_ready;
    assign lane_input_fire = lane_in_valid && lane_all_in_ready;
    assign conv_out_ready = lane_input_fire;

    assign lane_all_out_valid = &lane_out_valid;
    assign reduce_in_valid = (state_q == ST_LANE_WAIT) && lane_all_out_valid && (mode_q != MODE_SV_OUTER);
    assign reduce_input_fire = reduce_in_valid && reduce_in_ready;
    assign lane_out_ready = (mode_q == MODE_SV_OUTER) ? (state_q == ST_LANE_WAIT) : reduce_input_fire;
    assign lane_output_fire = (state_q == ST_LANE_WAIT) && lane_all_out_valid && lane_out_ready;

    assign reduce_out_ready = tile_add_input_fire;
    assign tile_add_in_valid = (state_q == ST_REDUCE_WAIT) && reduce_out_valid;
    assign tile_add_input_fire = tile_add_in_valid && tile_add_in_ready;
    assign reduce_output_fire = reduce_out_valid && reduce_out_ready;
    assign tile_add_out_ready = (state_q == ST_TILE_ADD_WAIT);
    assign tile_add_output_fire = tile_add_out_valid && tile_add_out_ready;

    assign bank_clear_valid = input_fire && (in_mode == MODE_SV_OUTER) && (in_clear || in_tile_first);
    assign bank_clear_mask = effective_lane_mask;
    assign bank_update_valid = lane_output_fire && (mode_q == MODE_SV_OUTER);
    assign bank_update_mask = lane_mask_q;

    accumulator_bank #(
        .PE_NUM(PE_NUM)
    ) u_accumulator_bank (
        .clk           (clk),
        .rst_n         (rst_n),
        .clear_valid   (bank_clear_valid),
        .clear_mask    (bank_clear_mask),
        .update_valid  (bank_update_valid),
        .update_mask   (bank_update_mask),
        .update_values (bank_update_values),
        .read_values   (bank_read_values)
    );

    genvar lane_g;
    generate
        for (lane_g = 0; lane_g < PE_NUM; lane_g++) begin : g_lanes
            wire lane_active_for_conv = lane_mask_q[lane_g];
            wire [15:0] conv_a_input = ((mode_q != MODE_SV_OUTER) && lane_active_for_conv) ?
                vector_a_q[lane_g*16 +: 16] : 16'd0;
            wire [15:0] conv_b_input = lane_active_for_conv ? vector_b_q[lane_g*16 +: 16] : 16'd0;

            fp16_to_fp32 #(
                .META_W(META_W),
                .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
            ) u_conv_a (
                .clk                  (clk),
                .rst_n                (rst_n),
                .in_valid             (conv_in_valid),
                .in_ready             (conv_a_in_ready[lane_g]),
                .in_data              (conv_a_input),
                .in_meta              (meta_q),
                .in_last              (last_q),
                .out_valid            (conv_a_out_valid[lane_g]),
                .out_ready            (conv_out_ready),
                .out_data             (conv_a_data[lane_g]),
                .out_meta             (),
                .out_last             (),
                .out_invalid          (conv_a_invalid[lane_g]),
                .out_underflow_or_ftz ()
            );

            fp16_to_fp32 #(
                .META_W(META_W),
                .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
            ) u_conv_b (
                .clk                  (clk),
                .rst_n                (rst_n),
                .in_valid             (conv_in_valid),
                .in_ready             (conv_b_in_ready[lane_g]),
                .in_data              (conv_b_input),
                .in_meta              (meta_q),
                .in_last              (last_q),
                .out_valid            (conv_b_out_valid[lane_g]),
                .out_ready            (conv_out_ready),
                .out_data             (conv_b_data[lane_g]),
                .out_meta             (),
                .out_last             (),
                .out_invalid          (conv_b_invalid[lane_g]),
                .out_underflow_or_ftz ()
            );

            pe_lane #(
                .META_W(META_W),
                .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
            ) u_pe_lane (
                .clk            (clk),
                .rst_n          (rst_n),
                .in_valid       (lane_in_valid),
                .in_ready       (lane_in_ready[lane_g]),
                .in_mode        ((mode_q == MODE_SV_OUTER) ? PE_MODE_FMA : PE_MODE_PRODUCT),
                .in_lane_enable (1'b1),
                .in_lane_mask   (lane_mask_q[lane_g]),
                .in_scalar      ((mode_q == MODE_SV_OUTER) ? scalar_q : conv_a_data[lane_g]),
                .in_vector      (conv_b_data[lane_g]),
                .in_accumulator (bank_read_values[lane_g*32 +: 32]),
                .in_meta        (meta_q),
                .in_last        (last_q),
                .out_valid      (lane_out_valid[lane_g]),
                .out_ready      (lane_out_ready),
                .out_result     (lane_result[lane_g]),
                .out_status     (lane_status[lane_g]),
                .out_invalid    (lane_invalid[lane_g]),
                .out_lane_active(lane_out_active[lane_g]),
                .out_meta       (),
                .out_last       ()
            );

            assign lane_result_packed[lane_g*32 +: 32] = lane_result[lane_g];
        end
    endgenerate

    always_comb begin
        bank_update_values = '0;
        for (int lane = 0; lane < PE_NUM; lane++) begin
            bank_update_values[lane*32 +: 32] =
                lane_mask_q[lane] ? lane_result[lane] : bank_read_values[lane*32 +: 32];
        end
    end

    fp32_reduction_tree #(
        .PE_NUM(PE_NUM),
        .META_W(META_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_reduction_tree (
        .clk          (clk),
        .rst_n        (rst_n),
        .in_valid     (reduce_in_valid),
        .in_ready     (reduce_in_ready),
        .in_values    (lane_result_packed),
        .in_lane_mask (lane_mask_q),
        .in_meta      (meta_q),
        .in_last      (last_q),
        .out_valid    (reduce_out_valid),
        .out_ready    (reduce_out_ready),
        .out_sum      (reduce_sum),
        .out_status   (reduce_status),
        .out_invalid  (reduce_invalid),
        .out_meta     (),
        .out_last     (),
        .busy         (reduce_busy)
    );

    fp32_add_wrapper #(
        .META_W(META_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_tile_accumulator_add (
        .clk         (clk),
        .rst_n       (rst_n),
        .in_valid    (tile_add_in_valid),
        .in_ready    (tile_add_in_ready),
        .in_a        (inner_acc_q),
        .in_b        (reduce_sum),
        .in_meta     (meta_q),
        .in_last     (last_q),
        .out_valid   (tile_add_out_valid),
        .out_ready   (tile_add_out_ready),
        .out_result  (tile_add_result),
        .out_status  (tile_add_status),
        .out_invalid (tile_add_invalid),
        .out_meta    (),
        .out_last    ()
    );

    pe_perf_counter #(
        .PE_NUM(PE_NUM),
        .COUNTER_W(COUNTER_W)
    ) u_perf_counter (
        .clk                   (clk),
        .rst_n                 (rst_n),
        .clear                 (1'b0),
        .busy                  ((state_q != ST_IDLE) || out_valid_q),
        .lane_op_fire          (lane_output_fire),
        .active_lanes          (active_lane_count),
        .input_stall           (in_valid && !in_ready),
        .output_stall          (out_valid && !out_ready),
        .mode_switch           (mode_switch_fire),
        .tile_fire             (input_fire),
        .operation_fire        (output_fire),
        .invalid_fire          (final_invalid_fire),
        .total_cycles          (perf_total_cycles),
        .busy_cycles           (perf_busy_cycles),
        .active_lane_cycles    (perf_active_lane_cycles),
        .available_lane_cycles (perf_available_lane_cycles),
        .input_stall_cycles    (perf_input_stall_cycles),
        .output_stall_cycles   (perf_output_stall_cycles),
        .mode_switch_cycles    (perf_mode_switch_cycles),
        .tile_count            (perf_tile_count),
        .operation_count       (perf_operation_count),
        .invalid_count         (perf_invalid_count)
    );

    always_comb begin
        mode_switch_fire = input_fire && last_mode_valid_q && (last_mode_q != in_mode);
        final_invalid_fire = output_fire && out_invalid_q;
    end

    always_comb begin
        lane_status_or_comb = 8'd0;
        lane_invalid_or_comb = 1'b0;
        outer_next_vector_comb = bank_read_values;
        for (int lane = 0; lane < PE_NUM; lane++) begin
            if (lane_mask_q[lane]) begin
                lane_status_or_comb = lane_status_or_comb | lane_status[lane];
                lane_invalid_or_comb = lane_invalid_or_comb | lane_invalid[lane] | conv_b_invalid[lane];
                if (mode_q != MODE_SV_OUTER) begin
                    lane_invalid_or_comb = lane_invalid_or_comb | conv_a_invalid[lane];
                end
                outer_next_vector_comb[lane*32 +: 32] = lane_result[lane];
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            mode_q <= MODE_GEMV;
            tile_first_q <= 1'b0;
            tile_last_q <= 1'b0;
            lane_mask_q <= '0;
            scalar_q <= 32'd0;
            vector_a_q <= '0;
            vector_b_q <= '0;
            meta_q <= '0;
            last_q <= 1'b0;
            inner_acc_q <= 32'd0;
            have_inner_partial_q <= 1'b0;
            have_outer_partial_q <= 1'b0;
            last_mode_q <= MODE_GEMV;
            last_mode_valid_q <= 1'b0;
            lane_status_or_q <= 8'd0;
            lane_invalid_or_q <= 1'b0;
            reduce_status_or_q <= 8'd0;
            reduce_invalid_or_q <= 1'b0;
            out_valid_q <= 1'b0;
            out_mode_q <= MODE_GEMV;
            out_scalar_q <= 32'd0;
            out_vector_q <= '0;
            out_lane_mask_q <= '0;
            out_status_q <= 8'd0;
            out_invalid_q <= 1'b0;
            out_meta_q <= '0;
            out_last_q <= 1'b0;
        end else begin
            if (out_valid_q && out_ready) begin
                out_valid_q <= 1'b0;
            end

            unique case (state_q)
                ST_IDLE: begin
                    if (input_fire) begin
                        mode_q <= in_mode;
                        tile_first_q <= in_tile_first;
                        tile_last_q <= in_tile_last;
                        lane_mask_q <= effective_lane_mask;
                        scalar_q <= in_scalar_fp32;
                        vector_a_q <= in_vector_a_fp16;
                        vector_b_q <= in_vector_b_fp16;
                        meta_q <= in_meta;
                        last_q <= in_last;
                        last_mode_q <= in_mode;
                        last_mode_valid_q <= 1'b1;

                        if ((in_mode != MODE_SV_OUTER) && (in_clear || in_tile_first)) begin
                            inner_acc_q <= 32'd0;
                            have_inner_partial_q <= 1'b0;
                        end
                        if ((in_mode == MODE_SV_OUTER) && (in_clear || in_tile_first)) begin
                            have_outer_partial_q <= 1'b0;
                        end
                        state_q <= ST_CONV_START;
                    end
                end

                ST_CONV_START: begin
                    if (conv_input_fire) begin
                        state_q <= ST_CONV_WAIT;
                    end
                end

                ST_CONV_WAIT: begin
                    if (lane_input_fire) begin
                        state_q <= ST_LANE_WAIT;
                    end
                end

                ST_LANE_WAIT: begin
                    if (lane_output_fire) begin
                        lane_status_or_q <= lane_status_or_comb;
                        lane_invalid_or_q <= lane_invalid_or_comb;
                        if (mode_q == MODE_SV_OUTER) begin
                            have_outer_partial_q <= 1'b1;
                            if (tile_last_q) begin
                                out_vector_q <= outer_next_vector_comb;
                                out_valid_q <= 1'b1;
                                out_mode_q <= mode_q;
                                out_scalar_q <= 32'd0;
                                out_lane_mask_q <= lane_mask_q;
                                out_status_q <= lane_status_or_comb;
                                out_invalid_q <= lane_invalid_or_comb;
                                out_meta_q <= meta_q;
                                out_last_q <= last_q;
                                have_outer_partial_q <= 1'b0;
                                state_q <= ST_OUTPUT;
                            end else begin
                                state_q <= ST_IDLE;
                            end
                        end else begin
                            state_q <= ST_REDUCE_WAIT;
                        end
                    end
                end

                ST_REDUCE_WAIT: begin
                    if (tile_add_input_fire) begin
                        reduce_status_or_q <= lane_status_or_q | reduce_status;
                        reduce_invalid_or_q <= lane_invalid_or_q | reduce_invalid;
                        state_q <= ST_TILE_ADD_WAIT;
                    end
                end

                ST_TILE_ADD_WAIT: begin
                    if (tile_add_output_fire) begin
                        inner_acc_q <= tile_add_result;
                        have_inner_partial_q <= !tile_last_q;
                        if (tile_last_q) begin
                            out_valid_q <= 1'b1;
                            out_mode_q <= mode_q;
                            out_scalar_q <= tile_add_result;
                            out_vector_q <= '0;
                            out_lane_mask_q <= lane_mask_q;
                            out_status_q <= reduce_status_or_q | tile_add_status;
                            out_invalid_q <= reduce_invalid_or_q | tile_add_invalid;
                            out_meta_q <= meta_q;
                            out_last_q <= last_q;
                            state_q <= ST_OUTPUT;
                        end else begin
                            state_q <= ST_IDLE;
                        end
                    end
                end

                ST_OUTPUT: begin
                    if (!out_valid_q || out_ready) begin
                        state_q <= ST_IDLE;
                    end
                end

                default: state_q <= ST_IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    logic [31:0] accepted_count;
    logic [31:0] emitted_count;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accepted_count <= 32'd0;
            emitted_count <= 32'd0;
        end else begin
            if (input_fire) begin
                accepted_count <= accepted_count + 32'd1;
            end
            if (output_fire) begin
                emitted_count <= emitted_count + 32'd1;
            end
        end
    end

    always_ff @(posedge clk) begin
        if (rst_n) begin
            assert ((in_mode == MODE_GEMV) || (in_mode == MODE_QK_INNER) || (in_mode == MODE_SV_OUTER))
                else $error("reconfigurable_pe_core unsupported mode");
            assert (!(input_fire && (effective_lane_mask == '0)))
                else $error("reconfigurable_pe_core lane_mask_legal failed");
            assert (!(input_fire && !in_tile_first && !have_inner_partial_q && (in_mode != MODE_SV_OUTER) && !in_clear))
                else $error("reconfigurable_pe_core tile_first/tile_last sequence legal failed");
            assert (!(input_fire && !in_tile_first && !have_outer_partial_q && (in_mode == MODE_SV_OUTER) && !in_clear))
                else $error("reconfigurable_pe_core outer clear/tile sequence legal failed");
            assert (!(out_valid && $isunknown({out_mode, out_scalar_fp32, out_vector_fp32, out_lane_mask, out_status, out_invalid, out_meta, out_last})))
                else $error("reconfigurable_pe_core no_unknown_result_when_valid failed");
            assert (accepted_count >= emitted_count)
                else $error("reconfigurable_pe_core transaction_count_conserved failed");

            if ($past(rst_n)) begin
                if ($past(in_valid && !in_ready)) begin
                    assert (in_valid)
                        else $error("reconfigurable_pe_core valid_stable_until_ready failed");
                    assert ($stable({in_mode, in_clear, in_tile_first, in_tile_last, in_use_explicit_mask,
                                     in_active_lanes, in_lane_mask, in_scalar_fp32,
                                     in_vector_a_fp16, in_vector_b_fp16}))
                        else $error("reconfigurable_pe_core payload_stable_until_ready failed");
                    assert ($stable(in_meta))
                        else $error("reconfigurable_pe_core metadata_stable_until_ready failed");
                    assert ($stable(in_last))
                        else $error("reconfigurable_pe_core last_stable_until_ready failed");
                end

                if ($past(out_valid && !out_ready)) begin
                    assert (out_valid)
                        else $error("reconfigurable_pe_core output_order_preserved failed");
                    assert ($stable({out_mode, out_scalar_fp32, out_vector_fp32, out_lane_mask, out_status, out_invalid, out_meta, out_last}))
                        else $error("reconfigurable_pe_core output payload changed under backpressure");
                end
            end

            assert (!((state_q != ST_IDLE) && input_fire && (in_mode != mode_q)))
                else $error("reconfigurable_pe_core no_mode_change_while_busy failed");
            assert (!(out_valid && !out_ready && input_fire && in_clear))
                else $error("reconfigurable_pe_core no_clear_while_result_pending failed");
        end
    end
`endif
endmodule

`default_nettype wire
