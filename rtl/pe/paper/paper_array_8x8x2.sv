`default_nettype none

module paper_array_8x8x2 #(
    parameter int META_W = 16,
    parameter int COUNTER_W = 64,
    parameter bit ASSERT_ON_INVALID = 1'b1
) (
    input  logic                         clk,
    input  logic                         rst_n,

    input  logic                         cmd_valid,
    output logic                         cmd_ready,
    input  logic [1:0]                   cmd_mode,
    input  logic [15:0]                  cmd_k_size,
    input  logic [15:0]                  cmd_m_size,
    input  logic [15:0]                  cmd_n_size,
    input  logic [15:0]                  cmd_tile_id,
    input  logic [META_W-1:0]            cmd_meta,
    input  logic                         cmd_clear_acc,
    input  logic                         cmd_tile_last,
    input  logic [1:0]                   cmd_group_mask,
    input  logic [127:0]                 cmd_lane_mask,
    input  logic [31:0]                  cmd_scalar_fp32,
    input  logic [128*16-1:0]            cmd_operand_a_fp16,
    input  logic [128*16-1:0]            cmd_operand_b_fp16,
    input  logic                         cmd_last,

    output logic                         result_valid,
    input  logic                         result_ready,
    output logic [1:0]                   result_mode,
    output logic [15:0]                  result_tile_id,
    output logic [31:0]                  result_scalar_fp32,
    output logic [128*32-1:0]            result_vector_fp32,
    output logic [127:0]                 result_lane_mask,
    output logic [7:0]                   result_status,
    output logic                         result_invalid,
    output logic [META_W-1:0]            result_meta,
    output logic                         result_last,

    output logic                         done_valid,
    input  logic                         done_ready,
    output logic [7:0]                   done_status,
    output logic                         done_invalid,
    output logic [META_W-1:0]            done_meta,

    output logic [COUNTER_W-1:0]         perf_paper_array_active_cycles,
    output logic [COUNTER_W-1:0]         perf_paper_array_idle_cycles,
    output logic [COUNTER_W-1:0]         perf_inner_mode_cycles,
    output logic [COUNTER_W-1:0]         perf_outer_mode_cycles,
    output logic [COUNTER_W-1:0]         perf_group0_active_cycles,
    output logic [COUNTER_W-1:0]         perf_group1_active_cycles,
    output logic [COUNTER_W-1:0]         perf_tail_masked_pe_cycles,
    output logic [COUNTER_W-1:0]         perf_mode_switch_cycles,
    output logic [COUNTER_W-1:0]         perf_array_input_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_array_output_stall_cycles
);
    localparam logic [1:0] MODE_INNER_PRODUCT = 2'd1;
    localparam logic [1:0] MODE_OUTER_PRODUCT = 2'd2;
    localparam int GROUP_COUNT = 2;
    localparam int ROW_COUNT = 8;
    localparam int COLUMN_COUNT = 8;
    localparam int PE_CELL_COUNT = 128;

    typedef enum logic [2:0] {
        ST_IDLE,
        ST_GROUP_SEND,
        ST_GROUP_WAIT,
        ST_COMBINE_WAIT,
        ST_RESULT,
        ST_DONE
    } state_e;

    state_e state_q;

    logic [1:0] mode_q;
    logic [15:0] tile_id_q;
    logic [META_W-1:0] meta_q;
    logic cmd_clear_acc_q;
    logic [31:0] cmd_scalar_q;
    logic tile_last_q;
    logic [1:0] group_mask_q;
    logic [127:0] lane_mask_q;
    logic [128*16-1:0] operand_a_q;
    logic [128*16-1:0] operand_b_q;
    logic last_q;
    logic [1:0] last_mode_q;
    logic last_mode_valid_q;

    logic [1:0] group_cmd_ready;
    logic [1:0] group_out_valid;
    logic [1:0] group_out_ready;
    logic [1:0] group_out_mode [0:1];
    logic [31:0] group_out_scalar [0:1];
    logic [64*32-1:0] group_out_vector [0:1];
    logic [63:0] group_out_lane_mask [0:1];
    logic [7:0] group_out_status [0:1];
    logic [1:0] group_out_invalid;
    logic [META_W-1:0] group_out_meta [0:1];
    logic [1:0] group_out_last;
    logic [COUNTER_W-1:0] group_active_cycles [0:1];
    logic [COUNTER_W-1:0] group_idle_cycles [0:1];

    logic combine_in_valid;
    logic combine_in_ready;
    logic combine_out_valid;
    logic combine_out_ready;
    logic [31:0] combine_sum;
    logic [7:0] combine_status;
    logic combine_invalid;

    logic result_valid_q;
    logic [1:0] result_mode_q;
    logic [15:0] result_tile_id_q;
    logic [31:0] result_scalar_q;
    logic [128*32-1:0] result_vector_q;
    logic [127:0] result_lane_mask_q;
    logic [7:0] result_status_q;
    logic result_invalid_q;
    logic [META_W-1:0] result_meta_q;
    logic result_last_q;

    logic done_valid_q;
    logic [7:0] done_status_q;
    logic done_invalid_q;
    logic [META_W-1:0] done_meta_q;

    logic [7:0] status_or_comb;
    logic invalid_or_comb;
    logic group_all_ready;
    logic group_all_valid;
    logic command_active;
    logic [7:0] l1_l2_group_mask;

    wire input_fire = cmd_valid && cmd_ready;
    wire result_fire = result_valid && result_ready;
    wire done_fire = done_valid && done_ready;
    wire group_input_fire = (state_q == ST_GROUP_SEND) && group_all_ready;
    wire group_output_fire = (state_q == ST_GROUP_WAIT) && group_all_valid && (&group_out_ready);
    wire combine_output_fire = combine_out_valid && combine_out_ready;

    assign command_active = (state_q != ST_IDLE);
    assign cmd_ready = (state_q == ST_IDLE) && !result_valid_q && !done_valid_q;
    assign group_all_ready = &group_cmd_ready;
    assign group_all_valid = &group_out_valid;
    assign combine_in_valid = (state_q == ST_GROUP_WAIT) && group_all_valid && (mode_q == MODE_INNER_PRODUCT);
    assign combine_out_ready = (state_q == ST_COMBINE_WAIT) && (!result_valid_q || result_ready);
    assign l1_l2_group_mask = {6'd0, group_mask_q};

    assign result_valid = result_valid_q;
    assign result_mode = result_mode_q;
    assign result_tile_id = result_tile_id_q;
    assign result_scalar_fp32 = result_scalar_q;
    assign result_vector_fp32 = result_vector_q;
    assign result_lane_mask = result_lane_mask_q;
    assign result_status = result_status_q;
    assign result_invalid = result_invalid_q;
    assign result_meta = result_meta_q;
    assign result_last = result_last_q;

    assign done_valid = done_valid_q;
    assign done_status = done_status_q;
    assign done_invalid = done_invalid_q;
    assign done_meta = done_meta_q;

    assign perf_group0_active_cycles = group_active_cycles[0];
    assign perf_group1_active_cycles = group_active_cycles[1];

    always_comb begin
        group_out_ready[0] = 1'b0;
        group_out_ready[1] = 1'b0;
        if (state_q == ST_GROUP_WAIT && group_all_valid) begin
            if (mode_q == MODE_INNER_PRODUCT) begin
                group_out_ready[0] = combine_in_ready;
                group_out_ready[1] = combine_in_ready;
            end else begin
                group_out_ready[0] = (!result_valid_q || result_ready);
                group_out_ready[1] = (!result_valid_q || result_ready);
            end
        end
    end

    always_comb begin
        status_or_comb = group_out_status[0] | group_out_status[1];
        invalid_or_comb = group_out_invalid[0] | group_out_invalid[1];
    end

    genvar group_g;
    generate
        for (group_g = 0; group_g < 2; group_g++) begin : g_groups
            paper_pe_group #(
                .GROUP_INDEX(group_g),
                .META_W(META_W),
                .COUNTER_W(COUNTER_W),
                .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
            ) u_group (
                .clk                       (clk),
                .rst_n                     (rst_n),
                .cmd_valid                 (state_q == ST_GROUP_SEND),
                .cmd_ready                 (group_cmd_ready[group_g]),
                .cmd_mode                  (mode_q),
                .cmd_group_active          (group_mask_q[group_g]),
                .cmd_clear_acc             (cmd_clear_acc_q),
                .cmd_tile_last             (tile_last_q),
                .cmd_lane_mask             (lane_mask_q[group_g*64 +: 64]),
                .cmd_scalar_fp32           (cmd_scalar_q),
                .cmd_operand_a_fp16        (operand_a_q[group_g*64*16 +: 64*16]),
                .cmd_operand_b_fp16        (operand_b_q[group_g*64*16 +: 64*16]),
                .cmd_meta                  (meta_q),
                .cmd_last                  (last_q),
                .out_valid                 (group_out_valid[group_g]),
                .out_ready                 (group_out_ready[group_g]),
                .out_mode                  (group_out_mode[group_g]),
                .out_scalar_fp32           (group_out_scalar[group_g]),
                .out_vector_fp32           (group_out_vector[group_g]),
                .out_lane_mask             (group_out_lane_mask[group_g]),
                .out_status                (group_out_status[group_g]),
                .out_invalid               (group_out_invalid[group_g]),
                .out_meta                  (group_out_meta[group_g]),
                .out_last                  (group_out_last[group_g]),
                .perf_group_active_cycles  (group_active_cycles[group_g]),
                .perf_group_idle_cycles    (group_idle_cycles[group_g])
            );
        end
    endgenerate

    paper_l2_reduction #(
        .META_W(META_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_group_combine (
        .clk            (clk),
        .rst_n          (rst_n),
        .in_valid       (combine_in_valid),
        .in_ready       (combine_in_ready),
        .in_values_fp32 ({192'd0, group_out_scalar[1], group_out_scalar[0]}),
        .in_mask        (l1_l2_group_mask),
        .in_meta        (meta_q),
        .in_last        (last_q),
        .out_valid      (combine_out_valid),
        .out_ready      (combine_out_ready),
        .out_sum_fp32   (combine_sum),
        .out_status     (combine_status),
        .out_invalid    (combine_invalid),
        .out_meta       (),
        .out_last       ()
    );

    function automatic [COUNTER_W-1:0] popcount_zeros(input logic [127:0] mask);
        logic [COUNTER_W-1:0] count;
        begin
            count = '0;
            for (int i = 0; i < 128; i++) begin
                if (!mask[i]) begin
                    count = count + COUNTER_W'(1);
                end
            end
            popcount_zeros = count;
        end
    endfunction

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            mode_q <= MODE_INNER_PRODUCT;
            tile_id_q <= 16'd0;
            meta_q <= '0;
            cmd_clear_acc_q <= 1'b0;
            cmd_scalar_q <= 32'd0;
            tile_last_q <= 1'b0;
            group_mask_q <= 2'd0;
            lane_mask_q <= 128'd0;
            operand_a_q <= '0;
            operand_b_q <= '0;
            last_q <= 1'b0;
            last_mode_q <= MODE_INNER_PRODUCT;
            last_mode_valid_q <= 1'b0;
            result_valid_q <= 1'b0;
            result_mode_q <= MODE_INNER_PRODUCT;
            result_tile_id_q <= 16'd0;
            result_scalar_q <= 32'd0;
            result_vector_q <= '0;
            result_lane_mask_q <= 128'd0;
            result_status_q <= 8'd0;
            result_invalid_q <= 1'b0;
            result_meta_q <= '0;
            result_last_q <= 1'b0;
            done_valid_q <= 1'b0;
            done_status_q <= 8'd0;
            done_invalid_q <= 1'b0;
            done_meta_q <= '0;
            perf_paper_array_active_cycles <= '0;
            perf_paper_array_idle_cycles <= '0;
            perf_inner_mode_cycles <= '0;
            perf_outer_mode_cycles <= '0;
            perf_tail_masked_pe_cycles <= '0;
            perf_mode_switch_cycles <= '0;
            perf_array_input_stall_cycles <= '0;
            perf_array_output_stall_cycles <= '0;
        end else begin
            if (result_fire) begin
                result_valid_q <= 1'b0;
            end
            if (done_fire) begin
                done_valid_q <= 1'b0;
            end

            if (command_active) begin
                perf_paper_array_active_cycles <= perf_paper_array_active_cycles + COUNTER_W'(1);
                if (mode_q == MODE_INNER_PRODUCT) begin
                    perf_inner_mode_cycles <= perf_inner_mode_cycles + COUNTER_W'(1);
                end else if (mode_q == MODE_OUTER_PRODUCT) begin
                    perf_outer_mode_cycles <= perf_outer_mode_cycles + COUNTER_W'(1);
                end
            end else begin
                perf_paper_array_idle_cycles <= perf_paper_array_idle_cycles + COUNTER_W'(1);
            end
            if (cmd_valid && !cmd_ready) begin
                perf_array_input_stall_cycles <= perf_array_input_stall_cycles + COUNTER_W'(1);
            end
            if ((result_valid && !result_ready) || (done_valid && !done_ready)) begin
                perf_array_output_stall_cycles <= perf_array_output_stall_cycles + COUNTER_W'(1);
            end

            unique case (state_q)
                ST_IDLE: begin
                    if (input_fire) begin
                        if (last_mode_valid_q && (last_mode_q != cmd_mode)) begin
                            perf_mode_switch_cycles <= perf_mode_switch_cycles + COUNTER_W'(1);
                        end
                        last_mode_q <= cmd_mode;
                        last_mode_valid_q <= 1'b1;
                        mode_q <= cmd_mode;
                        tile_id_q <= cmd_tile_id;
                        meta_q <= cmd_meta;
                        cmd_clear_acc_q <= cmd_clear_acc;
                        cmd_scalar_q <= cmd_scalar_fp32;
                        tile_last_q <= cmd_tile_last;
                        group_mask_q <= cmd_group_mask;
                        lane_mask_q <= cmd_lane_mask;
                        operand_a_q <= cmd_operand_a_fp16;
                        operand_b_q <= cmd_operand_b_fp16;
                        last_q <= cmd_last;
                        perf_tail_masked_pe_cycles <= perf_tail_masked_pe_cycles + popcount_zeros(cmd_lane_mask);
                        state_q <= ST_GROUP_SEND;
                    end
                end

                ST_GROUP_SEND: begin
                    if (group_input_fire) begin
                        state_q <= ST_GROUP_WAIT;
                    end
                end

                ST_GROUP_WAIT: begin
                    if (group_output_fire) begin
                        if (mode_q == MODE_INNER_PRODUCT) begin
                            state_q <= ST_COMBINE_WAIT;
                        end else begin
                            if (tile_last_q) begin
                                result_valid_q <= 1'b1;
                                result_mode_q <= mode_q;
                                result_tile_id_q <= tile_id_q;
                                result_scalar_q <= 32'd0;
                                result_vector_q <= {group_out_vector[1], group_out_vector[0]};
                                result_lane_mask_q <= lane_mask_q;
                                result_status_q <= status_or_comb;
                                result_invalid_q <= invalid_or_comb;
                                result_meta_q <= meta_q;
                                result_last_q <= last_q;
                            end
                            done_valid_q <= 1'b1;
                            done_status_q <= status_or_comb;
                            done_invalid_q <= invalid_or_comb;
                            done_meta_q <= meta_q;
                            state_q <= tile_last_q ? ST_RESULT : ST_DONE;
                        end
                    end
                end

                ST_COMBINE_WAIT: begin
                    if (combine_output_fire) begin
                        result_valid_q <= 1'b1;
                        result_mode_q <= mode_q;
                        result_tile_id_q <= tile_id_q;
                        result_scalar_q <= combine_sum;
                        result_vector_q <= '0;
                        result_lane_mask_q <= lane_mask_q;
                        result_status_q <= status_or_comb | combine_status;
                        result_invalid_q <= invalid_or_comb | combine_invalid;
                        result_meta_q <= meta_q;
                        result_last_q <= last_q;
                        done_valid_q <= 1'b1;
                        done_status_q <= status_or_comb | combine_status;
                        done_invalid_q <= invalid_or_comb | combine_invalid;
                        done_meta_q <= meta_q;
                        state_q <= ST_RESULT;
                    end
                end

                ST_RESULT: begin
                    if ((!result_valid_q || result_ready) && (!done_valid_q || done_ready)) begin
                        state_q <= ST_IDLE;
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

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n) begin
            assert (GROUP_COUNT == 2)
                else $error("paper_array_8x8x2 group_count_is_two failed");
            assert (ROW_COUNT == 8)
                else $error("paper_array_8x8x2 row_count_is_eight failed");
            assert (COLUMN_COUNT == 8)
                else $error("paper_array_8x8x2 column_count_is_eight failed");
            assert (PE_CELL_COUNT == 128)
                else $error("paper_array_8x8x2 array_has_exactly_128_pe_cells failed");
            assert (!(cmd_valid && cmd_ready && cmd_group_mask == 2'd0))
                else $error("paper_array_8x8x2 group mask zero failed");
            assert (!(cmd_valid && cmd_ready && cmd_lane_mask == 128'd0))
                else $error("paper_array_8x8x2 input_index_in_range zero lane mask failed");
            assert (!(cmd_valid && cmd_ready &&
                      !((cmd_mode == MODE_INNER_PRODUCT) || (cmd_mode == MODE_OUTER_PRODUCT))))
                else $error("paper_array_8x8x2 input mode illegal failed");
            assert (!(cmd_valid && !cmd_ready && command_active && cmd_mode != mode_q))
                else $error("paper_array_8x8x2 no_mode_switch_with_inflight_data failed");
            if ($past(rst_n) && $past(cmd_valid && !cmd_ready)) begin
                assert (cmd_valid)
                    else $error("paper_array_8x8x2 command_stable_until_ready valid failed");
                assert ($stable({cmd_mode, cmd_k_size, cmd_m_size, cmd_n_size, cmd_tile_id,
                                 cmd_meta, cmd_clear_acc, cmd_tile_last, cmd_group_mask,
                                 cmd_lane_mask, cmd_scalar_fp32, cmd_operand_a_fp16,
                                 cmd_operand_b_fp16, cmd_last}))
                    else $error("paper_array_8x8x2 command_stable_until_ready payload failed");
            end
            if ($past(rst_n) && $past(result_valid && !result_ready)) begin
                assert (result_valid)
                    else $error("paper_array_8x8x2 output_stable_until_ready valid failed");
                assert ($stable({result_mode, result_tile_id, result_scalar_fp32,
                                 result_vector_fp32, result_lane_mask, result_status,
                                 result_invalid, result_meta, result_last}))
                    else $error("paper_array_8x8x2 output_stable_until_ready payload failed");
            end
            if ($past(rst_n) && $past(done_valid && !done_ready)) begin
                assert (done_valid)
                    else $error("paper_array_8x8x2 done_stable_until_ready valid failed");
                assert ($stable({done_status, done_invalid, done_meta}))
                    else $error("paper_array_8x8x2 done_stable_until_ready payload failed");
            end
            assert (!(result_valid && $isunknown({result_mode, result_tile_id, result_scalar_fp32,
                                                  result_vector_fp32, result_lane_mask,
                                                  result_status, result_invalid,
                                                  result_meta, result_last})))
                else $error("paper_array_8x8x2 no_unknown_output_when_valid failed");
            assert (!(done_valid && $isunknown({done_status, done_invalid, done_meta})))
                else $error("paper_array_8x8x2 no_unknown_done_when_valid failed");
        end
    end
`endif
endmodule

`default_nettype wire
