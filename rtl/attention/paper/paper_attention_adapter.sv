`default_nettype none

module paper_attention_adapter #(
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
    input  logic [1:0]                   in_mode,
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
    output logic [COUNTER_W-1:0]         perf_invalid_count,

    output logic [COUNTER_W-1:0]         perf_paper_array_active_cycles,
    output logic [COUNTER_W-1:0]         perf_paper_array_idle_cycles,
    output logic [COUNTER_W-1:0]         perf_inner_mode_cycles,
    output logic [COUNTER_W-1:0]         perf_outer_mode_cycles,
    output logic [COUNTER_W-1:0]         perf_group0_active_cycles,
    output logic [COUNTER_W-1:0]         perf_group1_active_cycles,
    output logic [COUNTER_W-1:0]         perf_tail_masked_pe_cycles,
    output logic [COUNTER_W-1:0]         perf_array_input_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_array_output_stall_cycles
);
    localparam logic [1:0] MODE_QK_INNER = 2'd1;
    localparam logic [1:0] MODE_SV_OUTER = 2'd2;

    typedef enum logic [2:0] {
        ST_IDLE,
        ST_SEND_ARRAY,
        ST_WAIT_ARRAY,
        ST_ARRAY_COMPLETE,
        ST_ADD_SEND,
        ST_ADD_WAIT,
        ST_OUTPUT
    } state_e;

    state_e state_q;

    logic [1:0] mode_q;
    logic clear_acc_q;
    logic tile_last_q;
    logic [PE_NUM-1:0] lane_mask_q;
    logic [31:0] scalar_q;
    logic [PE_NUM*16-1:0] vector_a_q;
    logic [PE_NUM*16-1:0] vector_b_q;
    logic [META_W-1:0] meta_q;
    logic last_q;
    logic [31:0] inner_acc_q;
    logic [7:0] status_acc_q;
    logic invalid_acc_q;
    logic [1:0] last_mode_q;
    logic last_mode_valid_q;
    logic [15:0] tile_id_q;

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
    logic [127:0] array_lane_mask;
    logic [128*16-1:0] array_operand_a;
    logic [128*16-1:0] array_operand_b;
    logic [1:0] array_group_mask;
    logic [PE_NUM*32-1:0] narrowed_array_vector;
    logic [COUNTER_W-1:0] active_lane_count;
    logic [COUNTER_W-1:0] input_active_lane_count;

    logic array_cmd_valid;
    logic array_cmd_ready;
    logic array_result_valid;
    logic array_result_ready;
    logic [1:0] array_result_mode;
    logic [15:0] array_result_tile_id;
    logic [31:0] array_result_scalar_fp32;
    logic [128*32-1:0] array_result_vector_fp32;
    logic [127:0] array_result_lane_mask;
    logic [7:0] array_result_status;
    logic array_result_invalid;
    logic [META_W-1:0] array_result_meta;
    logic array_result_last;
    logic array_done_valid;
    logic array_done_ready;
    logic [7:0] array_done_status;
    logic array_done_invalid;
    logic [META_W-1:0] array_done_meta;
    logic [COUNTER_W-1:0] unused_paper_array_mode_switch_cycles;

    logic array_result_seen_q;
    logic array_done_seen_q;
    logic [31:0] array_scalar_q;
    logic [PE_NUM*32-1:0] array_vector_q;
    logic [PE_NUM-1:0] array_lane_mask_q;
    logic [7:0] array_status_q;
    logic array_invalid_q;
    logic [META_W-1:0] array_meta_q;
    logic array_last_q;

    logic add_in_valid;
    logic add_in_ready;
    logic add_out_valid;
    logic add_out_ready;
    logic [31:0] add_result;
    logic [7:0] add_status;
    logic add_invalid;

    wire input_fire = in_valid && in_ready;
    wire output_fire = out_valid && out_ready;
    wire array_cmd_fire = array_cmd_valid && array_cmd_ready;
    wire array_result_fire = array_result_valid && array_result_ready;
    wire array_done_fire = array_done_valid && array_done_ready;
    wire add_in_fire = add_in_valid && add_in_ready;
    wire add_out_fire = add_out_valid && add_out_ready;

    initial begin
        if (PE_NUM <= 0 || PE_NUM > 128) begin
            $fatal(1, "paper_attention_adapter PE_NUM must be in 1..128");
        end
        if ((PE_NUM & (PE_NUM - 1)) != 0) begin
            $fatal(1, "paper_attention_adapter PE_NUM must be a power of two");
        end
        if (META_W <= 0 || COUNTER_W <= 0) begin
            $fatal(1, "paper_attention_adapter widths must be positive");
        end
    end

    lane_mask_generator #(
        .PE_NUM(PE_NUM)
    ) u_mask_generator (
        .use_explicit_mask  (in_use_explicit_mask),
        .explicit_lane_mask (in_lane_mask),
        .active_lanes       (in_active_lanes),
        .lane_mask          (effective_lane_mask)
    );

    assign in_ready = (state_q == ST_IDLE) && !out_valid_q;
    assign out_valid = out_valid_q;
    assign out_mode = out_mode_q;
    assign out_scalar_fp32 = out_scalar_q;
    assign out_vector_fp32 = out_vector_q;
    assign out_lane_mask = out_lane_mask_q;
    assign out_status = out_status_q;
    assign out_invalid = out_invalid_q;
    assign out_meta = out_meta_q;
    assign out_last = out_last_q;

    assign array_cmd_valid = (state_q == ST_SEND_ARRAY);
    assign array_result_ready = (state_q == ST_WAIT_ARRAY);
    assign array_done_ready = (state_q == ST_WAIT_ARRAY);

    assign add_in_valid = (state_q == ST_ADD_SEND);
    assign add_out_ready = (state_q == ST_ADD_WAIT);

    always_comb begin
        array_lane_mask = 128'd0;
        array_operand_a = '0;
        array_operand_b = '0;
        array_group_mask = 2'b00;
        active_lane_count = '0;
        for (int lane = 0; lane < PE_NUM; lane++) begin
            array_lane_mask[lane] = lane_mask_q[lane];
            array_operand_a[lane*16 +: 16] = vector_a_q[lane*16 +: 16];
            array_operand_b[lane*16 +: 16] = vector_b_q[lane*16 +: 16];
            if (lane_mask_q[lane]) begin
                active_lane_count = active_lane_count + COUNTER_W'(1);
                if (lane < 64) begin
                    array_group_mask[0] = 1'b1;
                end else begin
                    array_group_mask[1] = 1'b1;
                end
            end
        end
    end

    always_comb begin
        input_active_lane_count = '0;
        for (int lane = 0; lane < PE_NUM; lane++) begin
            if (effective_lane_mask[lane]) begin
                input_active_lane_count = input_active_lane_count + COUNTER_W'(1);
            end
        end
    end

    always_comb begin
        narrowed_array_vector = '0;
        for (int lane = 0; lane < PE_NUM; lane++) begin
            narrowed_array_vector[lane*32 +: 32] = array_result_vector_fp32[lane*32 +: 32];
        end
    end

    paper_array_8x8x2 #(
        .META_W(META_W),
        .COUNTER_W(COUNTER_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_paper_array_8x8x2 (
        .clk                              (clk),
        .rst_n                            (rst_n),
        .cmd_valid                        (array_cmd_valid),
        .cmd_ready                        (array_cmd_ready),
        .cmd_mode                         (mode_q),
        .cmd_k_size                       (16'(PE_NUM)),
        .cmd_m_size                       (16'd1),
        .cmd_n_size                       (16'd1),
        .cmd_tile_id                      (tile_id_q),
        .cmd_meta                         (meta_q),
        .cmd_clear_acc                    (clear_acc_q),
        .cmd_tile_last                    (tile_last_q),
        .cmd_group_mask                   (array_group_mask),
        .cmd_lane_mask                    (array_lane_mask),
        .cmd_scalar_fp32                  (scalar_q),
        .cmd_operand_a_fp16               (array_operand_a),
        .cmd_operand_b_fp16               (array_operand_b),
        .cmd_last                         (last_q),
        .result_valid                     (array_result_valid),
        .result_ready                     (array_result_ready),
        .result_mode                      (array_result_mode),
        .result_tile_id                   (array_result_tile_id),
        .result_scalar_fp32               (array_result_scalar_fp32),
        .result_vector_fp32               (array_result_vector_fp32),
        .result_lane_mask                 (array_result_lane_mask),
        .result_status                    (array_result_status),
        .result_invalid                   (array_result_invalid),
        .result_meta                      (array_result_meta),
        .result_last                      (array_result_last),
        .done_valid                       (array_done_valid),
        .done_ready                       (array_done_ready),
        .done_status                      (array_done_status),
        .done_invalid                     (array_done_invalid),
        .done_meta                        (array_done_meta),
        .perf_paper_array_active_cycles   (perf_paper_array_active_cycles),
        .perf_paper_array_idle_cycles     (perf_paper_array_idle_cycles),
        .perf_inner_mode_cycles           (perf_inner_mode_cycles),
        .perf_outer_mode_cycles           (perf_outer_mode_cycles),
        .perf_group0_active_cycles        (perf_group0_active_cycles),
        .perf_group1_active_cycles        (perf_group1_active_cycles),
        .perf_tail_masked_pe_cycles       (perf_tail_masked_pe_cycles),
        .perf_mode_switch_cycles          (unused_paper_array_mode_switch_cycles),
        .perf_array_input_stall_cycles    (perf_array_input_stall_cycles),
        .perf_array_output_stall_cycles   (perf_array_output_stall_cycles)
    );

    fp32_add_wrapper #(
        .META_W(META_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_inner_tile_accumulator (
        .clk         (clk),
        .rst_n       (rst_n),
        .in_valid    (add_in_valid),
        .in_ready    (add_in_ready),
        .in_a        (inner_acc_q),
        .in_b        (array_scalar_q),
        .in_meta     (meta_q),
        .in_last     (last_q),
        .out_valid   (add_out_valid),
        .out_ready   (add_out_ready),
        .out_result  (add_result),
        .out_status  (add_status),
        .out_invalid (add_invalid),
        .out_meta    (),
        .out_last    ()
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            mode_q <= MODE_QK_INNER;
            clear_acc_q <= 1'b0;
            tile_last_q <= 1'b0;
            lane_mask_q <= '0;
            scalar_q <= 32'd0;
            vector_a_q <= '0;
            vector_b_q <= '0;
            meta_q <= '0;
            last_q <= 1'b0;
            inner_acc_q <= 32'd0;
            status_acc_q <= 8'd0;
            invalid_acc_q <= 1'b0;
            last_mode_q <= MODE_QK_INNER;
            last_mode_valid_q <= 1'b0;
            tile_id_q <= 16'd0;
            out_valid_q <= 1'b0;
            out_mode_q <= MODE_QK_INNER;
            out_scalar_q <= 32'd0;
            out_vector_q <= '0;
            out_lane_mask_q <= '0;
            out_status_q <= 8'd0;
            out_invalid_q <= 1'b0;
            out_meta_q <= '0;
            out_last_q <= 1'b0;
            array_result_seen_q <= 1'b0;
            array_done_seen_q <= 1'b0;
            array_scalar_q <= 32'd0;
            array_vector_q <= '0;
            array_lane_mask_q <= '0;
            array_status_q <= 8'd0;
            array_invalid_q <= 1'b0;
            array_meta_q <= '0;
            array_last_q <= 1'b0;
            perf_total_cycles <= '0;
            perf_busy_cycles <= '0;
            perf_active_lane_cycles <= '0;
            perf_available_lane_cycles <= '0;
            perf_input_stall_cycles <= '0;
            perf_output_stall_cycles <= '0;
            perf_mode_switch_cycles <= '0;
            perf_tile_count <= '0;
            perf_operation_count <= '0;
            perf_invalid_count <= '0;
        end else begin
            if (output_fire) begin
                out_valid_q <= 1'b0;
            end

            perf_total_cycles <= perf_total_cycles + COUNTER_W'(1);
            if (state_q != ST_IDLE || out_valid_q) begin
                perf_busy_cycles <= perf_busy_cycles + COUNTER_W'(1);
            end
            if (input_fire) begin
                perf_tile_count <= perf_tile_count + COUNTER_W'(1);
                perf_active_lane_cycles <= perf_active_lane_cycles + input_active_lane_count;
                perf_available_lane_cycles <= perf_available_lane_cycles + COUNTER_W'(PE_NUM);
                if (last_mode_valid_q && last_mode_q != in_mode) begin
                    perf_mode_switch_cycles <= perf_mode_switch_cycles + COUNTER_W'(1);
                end
            end
            if (in_valid && !in_ready) begin
                perf_input_stall_cycles <= perf_input_stall_cycles + COUNTER_W'(1);
            end
            if (out_valid && !out_ready) begin
                perf_output_stall_cycles <= perf_output_stall_cycles + COUNTER_W'(1);
            end

            unique case (state_q)
                ST_IDLE: begin
                    if (input_fire) begin
                        mode_q <= in_mode;
                        clear_acc_q <= in_clear || in_tile_first;
                        tile_last_q <= in_tile_last;
                        lane_mask_q <= effective_lane_mask;
                        scalar_q <= in_scalar_fp32;
                        vector_a_q <= in_vector_a_fp16;
                        vector_b_q <= in_vector_b_fp16;
                        meta_q <= in_meta;
                        last_q <= in_last;
                        last_mode_q <= in_mode;
                        last_mode_valid_q <= 1'b1;
                        tile_id_q <= tile_id_q + 16'd1;
                        array_result_seen_q <= 1'b0;
                        array_done_seen_q <= 1'b0;
                        array_status_q <= 8'd0;
                        array_invalid_q <= 1'b0;
                        if (in_clear || in_tile_first) begin
                            inner_acc_q <= 32'd0;
                            status_acc_q <= 8'd0;
                            invalid_acc_q <= 1'b0;
                        end
                        state_q <= ST_SEND_ARRAY;
                    end
                end

                ST_SEND_ARRAY: begin
                    if (array_cmd_fire) begin
                        state_q <= ST_WAIT_ARRAY;
                    end
                end

                ST_WAIT_ARRAY: begin
                    if (array_result_fire) begin
                        array_result_seen_q <= 1'b1;
                        array_scalar_q <= array_result_scalar_fp32;
                        array_vector_q <= narrowed_array_vector;
                        array_lane_mask_q <= array_result_lane_mask[PE_NUM-1:0];
                        array_status_q <= array_status_q | array_result_status;
                        array_invalid_q <= array_invalid_q | array_result_invalid;
                        array_meta_q <= array_result_meta;
                        array_last_q <= array_result_last;
                    end
                    if (array_done_fire) begin
                        array_done_seen_q <= 1'b1;
                        array_status_q <= array_status_q | array_done_status;
                        array_invalid_q <= array_invalid_q | array_done_invalid;
                    end
                    if ((array_done_seen_q || array_done_fire) &&
                        ((mode_q == MODE_QK_INNER && (array_result_seen_q || array_result_fire)) ||
                         (mode_q == MODE_SV_OUTER && (!tile_last_q || array_result_seen_q || array_result_fire)))) begin
                        state_q <= ST_ARRAY_COMPLETE;
                    end
                end

                ST_ARRAY_COMPLETE: begin
                    status_acc_q <= status_acc_q | array_status_q;
                    invalid_acc_q <= invalid_acc_q | array_invalid_q;
                    if (mode_q == MODE_QK_INNER) begin
                        state_q <= ST_ADD_SEND;
                    end else if (tile_last_q) begin
                        out_valid_q <= 1'b1;
                        out_mode_q <= mode_q;
                        out_scalar_q <= 32'd0;
                        out_vector_q <= array_vector_q;
                        out_lane_mask_q <= array_lane_mask_q;
                        out_status_q <= status_acc_q | array_status_q;
                        out_invalid_q <= invalid_acc_q | array_invalid_q;
                        out_meta_q <= array_meta_q;
                        out_last_q <= array_last_q;
                        perf_operation_count <= perf_operation_count + COUNTER_W'(1);
                        if (invalid_acc_q | array_invalid_q) begin
                            perf_invalid_count <= perf_invalid_count + COUNTER_W'(1);
                        end
                        state_q <= ST_OUTPUT;
                    end else begin
                        state_q <= ST_IDLE;
                    end
                end

                ST_ADD_SEND: begin
                    if (add_in_fire) begin
                        state_q <= ST_ADD_WAIT;
                    end
                end

                ST_ADD_WAIT: begin
                    if (add_out_fire) begin
                        inner_acc_q <= add_result;
                        status_acc_q <= status_acc_q | add_status;
                        invalid_acc_q <= invalid_acc_q | add_invalid;
                        if (tile_last_q) begin
                            out_valid_q <= 1'b1;
                            out_mode_q <= mode_q;
                            out_scalar_q <= add_result;
                            out_vector_q <= '0;
                            out_lane_mask_q <= lane_mask_q;
                            out_status_q <= status_acc_q | add_status;
                            out_invalid_q <= invalid_acc_q | add_invalid;
                            out_meta_q <= meta_q;
                            out_last_q <= last_q;
                            perf_operation_count <= perf_operation_count + COUNTER_W'(1);
                            if (invalid_acc_q | add_invalid) begin
                                perf_invalid_count <= perf_invalid_count + COUNTER_W'(1);
                            end
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
    always_ff @(posedge clk) begin
        if (rst_n) begin
            assert ((in_mode == MODE_QK_INNER) || (in_mode == MODE_SV_OUTER))
                else $error("paper_attention_adapter mode supported failed");
            assert (!(input_fire && (effective_lane_mask == '0)))
                else $error("paper_attention_adapter lane_mask_legal failed");
            if ($past(rst_n) && $past(in_valid && !in_ready)) begin
                assert (in_valid)
                    else $error("paper_attention_adapter input valid stable failed");
                assert ($stable({in_mode, in_clear, in_tile_first, in_tile_last,
                                 in_use_explicit_mask, in_active_lanes, in_lane_mask,
                                 in_scalar_fp32, in_vector_a_fp16, in_vector_b_fp16,
                                 in_meta, in_last}))
                    else $error("paper_attention_adapter input payload stable failed");
            end
            if ($past(rst_n) && $past(out_valid && !out_ready)) begin
                assert (out_valid)
                    else $error("paper_attention_adapter output valid stable failed");
                assert ($stable({out_mode, out_scalar_fp32, out_vector_fp32,
                                 out_lane_mask, out_status, out_invalid,
                                 out_meta, out_last}))
                    else $error("paper_attention_adapter output payload stable failed");
            end
            assert (!(out_valid && $isunknown({out_mode, out_scalar_fp32,
                                               out_vector_fp32, out_lane_mask,
                                               out_status, out_invalid,
                                               out_meta, out_last})))
                else $error("paper_attention_adapter no_unknown_output_when_valid failed");
        end
    end
`endif
endmodule

`default_nettype wire
