`default_nettype none

module paper_pe_group #(
    parameter int GROUP_INDEX = 0,
    parameter int META_W = 16,
    parameter int COUNTER_W = 64,
    parameter bit ASSERT_ON_INVALID = 1'b1
) (
    input  logic                       clk,
    input  logic                       rst_n,

    input  logic                       cmd_valid,
    output logic                       cmd_ready,
    input  logic [1:0]                 cmd_mode,
    input  logic                       cmd_group_active,
    input  logic                       cmd_clear_acc,
    input  logic                       cmd_tile_last,
    input  logic [63:0]                cmd_lane_mask,
    input  logic [31:0]                cmd_scalar_fp32,
    input  logic [64*16-1:0]           cmd_operand_a_fp16,
    input  logic [64*16-1:0]           cmd_operand_b_fp16,
    input  logic [META_W-1:0]          cmd_meta,
    input  logic                       cmd_last,

    output logic                       out_valid,
    input  logic                       out_ready,
    output logic [1:0]                 out_mode,
    output logic [31:0]                out_scalar_fp32,
    output logic [64*32-1:0]           out_vector_fp32,
    output logic [63:0]                out_lane_mask,
    output logic [7:0]                 out_status,
    output logic                       out_invalid,
    output logic [META_W-1:0]          out_meta,
    output logic                       out_last,

    output logic [COUNTER_W-1:0]       perf_group_active_cycles,
    output logic [COUNTER_W-1:0]       perf_group_idle_cycles
);
    localparam logic [1:0] MODE_INNER_PRODUCT = 2'd1;
    localparam logic [1:0] MODE_OUTER_PRODUCT = 2'd2;

    typedef enum logic [2:0] {
        ST_IDLE,
        ST_CELL_SEND,
        ST_CELL_WAIT,
        ST_L1_WAIT,
        ST_L2_WAIT,
        ST_OUTPUT
    } state_e;

    state_e state_q;

    logic [1:0] mode_q;
    logic group_active_q;
    logic tile_last_q;
    logic [63:0] lane_mask_q;
    logic [31:0] scalar_q;
    logic [64*16-1:0] operand_a_q;
    logic [64*16-1:0] operand_b_q;
    logic [META_W-1:0] meta_q;
    logic last_q;
    logic cmd_clear_acc_q;

    logic [63:0] cell_op_ready;
    logic [63:0] cell_out_valid;
    logic [63:0] cell_out_ready;
    logic [64*32-1:0] cell_product_flat;
    logic [64*32-1:0] cell_acc_flat;
    logic [64*32-1:0] cell_forward_flat;
    logic [64*8-1:0] cell_status_flat;
    logic [63:0] cell_invalid;
    logic [63:0] cell_active_out;

    logic [7:0] l1_in_valid;
    logic [7:0] l1_in_ready;
    logic [7:0] l1_out_valid;
    logic [7:0] l1_out_ready;
    logic [8*32-1:0] l1_sums_flat;
    logic [8*8-1:0] l1_status_flat;
    logic [7:0] l1_invalid;

    logic l2_in_valid;
    logic l2_in_ready;
    logic l2_out_valid;
    logic l2_out_ready;
    logic [31:0] l2_sum;
    logic [7:0] l2_status;
    logic l2_invalid;

    logic out_valid_q;
    logic [1:0] out_mode_q;
    logic [31:0] out_scalar_q;
    logic [64*32-1:0] out_vector_q;
    logic [63:0] out_lane_mask_q;
    logic [7:0] out_status_q;
    logic out_invalid_q;
    logic [META_W-1:0] out_meta_q;
    logic out_last_q;

    logic [7:0] row_active_mask_comb;
    logic [7:0] status_or_comb;
    logic [7:0] l1_status_or_comb;
    logic invalid_or_comb;
    logic l1_all_ready;
    logic l1_all_valid;
    logic cell_all_ready;
    logic cell_all_valid;

    wire input_fire = cmd_valid && cmd_ready;
    wire output_fire = out_valid && out_ready;
    wire cell_input_fire = (state_q == ST_CELL_SEND) && cell_all_ready;
    wire cell_output_fire = (state_q == ST_CELL_WAIT) && cell_all_valid && (&cell_out_ready);
    wire l1_output_fire = (state_q == ST_L1_WAIT) && l1_all_valid && l2_in_ready;
    wire l2_output_fire = l2_out_valid && l2_out_ready;

    assign cmd_ready = (state_q == ST_IDLE) && !out_valid_q;
    assign out_valid = out_valid_q;
    assign out_mode = out_mode_q;
    assign out_scalar_fp32 = out_scalar_q;
    assign out_vector_fp32 = out_vector_q;
    assign out_lane_mask = out_lane_mask_q;
    assign out_status = out_status_q;
    assign out_invalid = out_invalid_q;
    assign out_meta = out_meta_q;
    assign out_last = out_last_q;

    assign cell_all_ready = &cell_op_ready;
    assign cell_all_valid = &cell_out_valid;
    assign l1_all_ready = &l1_in_ready;
    assign l1_all_valid = &l1_out_valid;
    assign l2_in_valid = (state_q == ST_L1_WAIT) && l1_all_valid;
    assign l2_out_ready = (state_q == ST_L2_WAIT) && (!out_valid_q || out_ready);

    always_comb begin
        for (int row = 0; row < 8; row++) begin
            l1_in_valid[row] = (state_q == ST_CELL_WAIT) && cell_all_valid &&
                               (mode_q == MODE_INNER_PRODUCT) && l1_all_ready;
            l1_out_ready[row] = (state_q == ST_L1_WAIT) && l2_in_ready;
        end
    end

    always_comb begin
        for (int lane = 0; lane < 64; lane++) begin
            cell_out_ready[lane] = 1'b0;
            if (state_q == ST_CELL_WAIT && cell_all_valid) begin
                if (mode_q == MODE_OUTER_PRODUCT) begin
                    cell_out_ready[lane] = 1'b1;
                end else begin
                    cell_out_ready[lane] = l1_all_ready;
                end
            end
        end
    end

    always_comb begin
        row_active_mask_comb = 8'd0;
        for (int row = 0; row < 8; row++) begin
            for (int col = 0; col < 8; col++) begin
                if (lane_mask_q[row*8 + col]) begin
                    row_active_mask_comb[row] = 1'b1;
                end
            end
        end
    end

    always_comb begin
        status_or_comb = 8'd0;
        l1_status_or_comb = 8'd0;
        invalid_or_comb = 1'b0;
        for (int lane = 0; lane < 64; lane++) begin
            status_or_comb = status_or_comb | cell_status_flat[lane*8 +: 8];
            invalid_or_comb = invalid_or_comb | cell_invalid[lane];
        end
        for (int row = 0; row < 8; row++) begin
            l1_status_or_comb = l1_status_or_comb | l1_status_flat[row*8 +: 8];
        end
    end

    genvar row_g;
    genvar col_g;
    generate
        for (row_g = 0; row_g < 8; row_g++) begin : g_rows
            for (col_g = 0; col_g < 8; col_g++) begin : g_cols
                localparam int LANE_INDEX = row_g * 8 + col_g;
                localparam int PE_TYPE_PARAM = (col_g % 2 == 0) ? 0 : 1;
                paper_pe_cell #(
                    .GROUP_INDEX(GROUP_INDEX),
                    .ROW_INDEX(row_g),
                    .COLUMN_INDEX(col_g),
                    .PE_TYPE(PE_TYPE_PARAM),
                    .META_W(META_W),
                    .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
                ) u_cell (
                    .clk                         (clk),
                    .rst_n                       (rst_n),
                    .op_valid                    (state_q == ST_CELL_SEND),
                    .op_ready                    (cell_op_ready[LANE_INDEX]),
                    .op_mode                     (mode_q),
                    .op_active                   (group_active_q && lane_mask_q[LANE_INDEX]),
                    .op_clear_acc                (cmd_clear_acc_q),
                    .op_operand_a_fp16           (operand_a_q[LANE_INDEX*16 +: 16]),
                    .op_operand_b_fp16           (operand_b_q[LANE_INDEX*16 +: 16]),
                    .op_scalar_fp32              (scalar_q),
                    .op_partial_sum_in           (32'd0),
                    .op_meta                     (meta_q),
                    .op_last                     (last_q),
                    .out_valid                   (cell_out_valid[LANE_INDEX]),
                    .out_ready                   (cell_out_ready[LANE_INDEX]),
                    .out_product_fp32            (cell_product_flat[LANE_INDEX*32 +: 32]),
                    .out_accumulator_fp32        (cell_acc_flat[LANE_INDEX*32 +: 32]),
                    .out_forwarded_partial_fp32  (cell_forward_flat[LANE_INDEX*32 +: 32]),
                    .out_status                  (cell_status_flat[LANE_INDEX*8 +: 8]),
                    .out_invalid                 (cell_invalid[LANE_INDEX]),
                    .out_active                  (cell_active_out[LANE_INDEX]),
                    .out_meta                    (),
                    .out_last                    ()
                );
            end

            paper_l1_reduction #(
                .META_W(META_W),
                .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
            ) u_l1 (
                .clk            (clk),
                .rst_n          (rst_n),
                .in_valid       (l1_in_valid[row_g]),
                .in_ready       (l1_in_ready[row_g]),
                .in_values_fp32 (cell_product_flat[row_g*8*32 +: 8*32]),
                .in_mask        (lane_mask_q[row_g*8 +: 8]),
                .in_meta        (meta_q),
                .in_last        (last_q),
                .out_valid      (l1_out_valid[row_g]),
                .out_ready      (l1_out_ready[row_g]),
                .out_sum_fp32   (l1_sums_flat[row_g*32 +: 32]),
                .out_status     (l1_status_flat[row_g*8 +: 8]),
                .out_invalid    (l1_invalid[row_g]),
                .out_meta       (),
                .out_last       ()
            );
        end
    endgenerate

    paper_l2_reduction #(
        .META_W(META_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_l2 (
        .clk            (clk),
        .rst_n          (rst_n),
        .in_valid       (l2_in_valid),
        .in_ready       (l2_in_ready),
        .in_values_fp32 (l1_sums_flat),
        .in_mask        (row_active_mask_comb),
        .in_meta        (meta_q),
        .in_last        (last_q),
        .out_valid      (l2_out_valid),
        .out_ready      (l2_out_ready),
        .out_sum_fp32   (l2_sum),
        .out_status     (l2_status),
        .out_invalid    (l2_invalid),
        .out_meta       (),
        .out_last       ()
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            mode_q <= MODE_INNER_PRODUCT;
            group_active_q <= 1'b0;
            cmd_clear_acc_q <= 1'b0;
            tile_last_q <= 1'b0;
            lane_mask_q <= 64'd0;
            scalar_q <= 32'd0;
            operand_a_q <= '0;
            operand_b_q <= '0;
            meta_q <= '0;
            last_q <= 1'b0;
            out_valid_q <= 1'b0;
            out_mode_q <= MODE_INNER_PRODUCT;
            out_scalar_q <= 32'd0;
            out_vector_q <= '0;
            out_lane_mask_q <= 64'd0;
            out_status_q <= 8'd0;
            out_invalid_q <= 1'b0;
            out_meta_q <= '0;
            out_last_q <= 1'b0;
            perf_group_active_cycles <= '0;
            perf_group_idle_cycles <= '0;
        end else begin
            if (output_fire) begin
                out_valid_q <= 1'b0;
            end

            if (state_q != ST_IDLE) begin
                perf_group_active_cycles <= perf_group_active_cycles + COUNTER_W'(1);
            end else begin
                perf_group_idle_cycles <= perf_group_idle_cycles + COUNTER_W'(1);
            end

            unique case (state_q)
                ST_IDLE: begin
                    if (input_fire) begin
                        mode_q <= cmd_mode;
                        group_active_q <= cmd_group_active;
                        cmd_clear_acc_q <= cmd_clear_acc;
                        tile_last_q <= cmd_tile_last;
                        lane_mask_q <= cmd_lane_mask;
                        scalar_q <= cmd_scalar_fp32;
                        operand_a_q <= cmd_operand_a_fp16;
                        operand_b_q <= cmd_operand_b_fp16;
                        meta_q <= cmd_meta;
                        last_q <= cmd_last;
                        state_q <= ST_CELL_SEND;
                    end
                end

                ST_CELL_SEND: begin
                    if (cell_input_fire) begin
                        state_q <= ST_CELL_WAIT;
                    end
                end

                ST_CELL_WAIT: begin
                    if (cell_output_fire) begin
                        if (mode_q == MODE_OUTER_PRODUCT) begin
                            out_valid_q <= 1'b1;
                            out_mode_q <= mode_q;
                            out_scalar_q <= 32'd0;
                            out_vector_q <= cell_acc_flat;
                            out_lane_mask_q <= lane_mask_q;
                            out_status_q <= status_or_comb;
                            out_invalid_q <= invalid_or_comb;
                            out_meta_q <= meta_q;
                            out_last_q <= last_q;
                            state_q <= ST_OUTPUT;
                        end else begin
                            state_q <= ST_L1_WAIT;
                        end
                    end
                end

                ST_L1_WAIT: begin
                    if (l1_output_fire) begin
                        state_q <= ST_L2_WAIT;
                    end
                end

                ST_L2_WAIT: begin
                    if (l2_output_fire) begin
                        out_valid_q <= 1'b1;
                        out_mode_q <= mode_q;
                        out_scalar_q <= l2_sum;
                        out_vector_q <= '0;
                        out_lane_mask_q <= lane_mask_q;
                        out_status_q <= status_or_comb | l1_status_or_comb | l2_status;
                        out_invalid_q <= invalid_or_comb | l2_invalid | (|l1_invalid);
                        out_meta_q <= meta_q;
                        out_last_q <= last_q;
                        state_q <= ST_OUTPUT;
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
            assert (!(cmd_valid && cmd_ready && cmd_group_active && cmd_lane_mask == 64'd0))
                else $error("paper_pe_group input_index_in_range lane mask failed");
            if ($past(rst_n) && $past(out_valid && !out_ready)) begin
                assert (out_valid)
                    else $error("paper_pe_group output_stable_until_ready valid failed");
                assert ($stable({out_mode, out_scalar_fp32, out_vector_fp32, out_lane_mask,
                                 out_status, out_invalid, out_meta, out_last}))
                    else $error("paper_pe_group output_stable_until_ready payload failed");
            end
            assert (!(out_valid && $isunknown({out_mode, out_scalar_fp32, out_vector_fp32,
                                               out_lane_mask, out_status, out_invalid,
                                               out_meta, out_last})))
                else $error("paper_pe_group no_unknown_output_when_valid failed");
        end
    end
`endif
endmodule

`default_nettype wire
