`default_nettype none

module fp32_reduction_tree #(
    parameter int PE_NUM = 8,
    parameter int META_W = 1,
    parameter bit ASSERT_ON_INVALID = 1'b1,
    localparam int LEVEL_W = (PE_NUM <= 1) ? 1 : $clog2(PE_NUM + 1),
    localparam int PAIR_W = (PE_NUM <= 2) ? 1 : $clog2(PE_NUM)
) (
    input  logic                   clk,
    input  logic                   rst_n,

    input  logic                   in_valid,
    output logic                   in_ready,
    input  logic [PE_NUM*32-1:0]   in_values,
    input  logic [PE_NUM-1:0]      in_lane_mask,
    input  logic [META_W-1:0]      in_meta,
    input  logic                   in_last,

    output logic                   out_valid,
    input  logic                   out_ready,
    output logic [31:0]            out_sum,
    output logic [7:0]             out_status,
    output logic                   out_invalid,
    output logic [META_W-1:0]      out_meta,
    output logic                   out_last,
    output logic                   busy
);
    typedef enum logic [1:0] {
        ST_IDLE,
        ST_ADD_START,
        ST_ADD_WAIT,
        ST_OUTPUT
    } state_e;

    state_e state_q;
    logic [31:0] value_q [0:PE_NUM-1];
    logic [7:0] status_q [0:PE_NUM-1];
    logic invalid_q [0:PE_NUM-1];
    logic [META_W-1:0] meta_q;
    logic last_q;
    logic [LEVEL_W-1:0] width_q;
    logic [PAIR_W-1:0] pair_q;
    logic [7:0] pending_child_status_q;
    logic pending_child_invalid_q;

    logic add_in_valid;
    logic add_in_ready;
    logic [31:0] add_in_a;
    logic [31:0] add_in_b;
    logic add_out_valid;
    logic add_out_ready;
    logic [31:0] add_out_result;
    logic [7:0] add_out_status;
    logic add_out_invalid;
    logic [META_W-1:0] add_out_meta;
    logic add_out_last;

    initial begin
        if (PE_NUM <= 0) begin
            $fatal(1, "fp32_reduction_tree PE_NUM must be positive");
        end
        if ((PE_NUM & (PE_NUM - 1)) != 0) begin
            $fatal(1, "fp32_reduction_tree PE_NUM must be a power of two");
        end
        if (META_W <= 0) begin
            $fatal(1, "fp32_reduction_tree META_W must be positive");
        end
    end

    assign in_ready = (state_q == ST_IDLE);
    assign busy = (state_q != ST_IDLE);
    assign add_in_valid = (state_q == ST_ADD_START);
    assign add_in_a = value_q[pair_q * 2];
    assign add_in_b = value_q[(pair_q * 2) + 1];
    assign add_out_ready = (state_q == ST_ADD_WAIT);

    fp32_add_wrapper #(
        .META_W(META_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_add (
        .clk         (clk),
        .rst_n       (rst_n),
        .in_valid    (add_in_valid),
        .in_ready    (add_in_ready),
        .in_a        (add_in_a),
        .in_b        (add_in_b),
        .in_meta     (meta_q),
        .in_last     (last_q),
        .out_valid   (add_out_valid),
        .out_ready   (add_out_ready),
        .out_result  (add_out_result),
        .out_status  (add_out_status),
        .out_invalid (add_out_invalid),
        .out_meta    (add_out_meta),
        .out_last    (add_out_last)
    );

    assign out_valid = (state_q == ST_OUTPUT);
    assign out_sum = value_q[0];
    assign out_status = status_q[0];
    assign out_invalid = invalid_q[0];
    assign out_meta = meta_q;
    assign out_last = last_q;

    wire input_fire = in_valid && in_ready;
    wire add_input_fire = add_in_valid && add_in_ready;
    wire add_output_fire = add_out_valid && add_out_ready;
    wire output_fire = out_valid && out_ready;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            meta_q <= '0;
            last_q <= 1'b0;
            width_q <= LEVEL_W'(PE_NUM);
            pair_q <= '0;
            pending_child_status_q <= '0;
            pending_child_invalid_q <= 1'b0;
            for (int lane = 0; lane < PE_NUM; lane++) begin
                value_q[lane] <= 32'd0;
                status_q[lane] <= 8'd0;
                invalid_q[lane] <= 1'b0;
            end
        end else begin
            unique case (state_q)
                ST_IDLE: begin
                    if (input_fire) begin
                        meta_q <= in_meta;
                        last_q <= in_last;
                        width_q <= LEVEL_W'(PE_NUM);
                        pair_q <= '0;
                        pending_child_status_q <= '0;
                        pending_child_invalid_q <= 1'b0;
                        for (int lane = 0; lane < PE_NUM; lane++) begin
                            value_q[lane] <= in_lane_mask[lane] ? in_values[lane*32 +: 32] : 32'd0;
                            status_q[lane] <= 8'd0;
                            invalid_q[lane] <= 1'b0;
                        end
                        if (PE_NUM == 1) begin
                            state_q <= ST_OUTPUT;
                        end else begin
                            state_q <= ST_ADD_START;
                        end
                    end
                end

                ST_ADD_START: begin
                    if (add_input_fire) begin
                        pending_child_status_q <= status_q[pair_q * 2] | status_q[(pair_q * 2) + 1];
                        pending_child_invalid_q <= invalid_q[pair_q * 2] | invalid_q[(pair_q * 2) + 1];
                        state_q <= ST_ADD_WAIT;
                    end
                end

                ST_ADD_WAIT: begin
                    if (add_output_fire) begin
                        value_q[pair_q] <= add_out_result;
                        status_q[pair_q] <= pending_child_status_q | add_out_status;
                        invalid_q[pair_q] <= pending_child_invalid_q | add_out_invalid;
                        if (pair_q == PAIR_W'((width_q >> 1) - 1)) begin
                            pair_q <= '0;
                            width_q <= width_q >> 1;
                            if ((width_q >> 1) == LEVEL_W'(1)) begin
                                state_q <= ST_OUTPUT;
                            end else begin
                                state_q <= ST_ADD_START;
                            end
                        end else begin
                            pair_q <= pair_q + PAIR_W'(1);
                            state_q <= ST_ADD_START;
                        end
                    end
                end

                ST_OUTPUT: begin
                    if (output_fire) begin
                        state_q <= ST_IDLE;
                    end
                end

                default: state_q <= ST_IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    logic add_inflight_q;
    logic [PAIR_W-1:0] add_launch_pair_q;
    logic [LEVEL_W-1:0] add_launch_width_q;
    logic [31:0] add_launch_a_q;
    logic [31:0] add_launch_b_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            add_inflight_q <= 1'b0;
            add_launch_pair_q <= '0;
            add_launch_width_q <= '0;
            add_launch_a_q <= 32'd0;
            add_launch_b_q <= 32'd0;
        end else begin
            if (add_input_fire) begin
                add_inflight_q <= 1'b1;
                add_launch_pair_q <= pair_q;
                add_launch_width_q <= width_q;
                add_launch_a_q <= add_in_a;
                add_launch_b_q <= add_in_b;
            end
            if (add_output_fire) begin
                add_inflight_q <= 1'b0;
            end
        end
    end

    always_ff @(posedge clk) begin
        if (rst_n) begin
            assert (!(out_valid && $isunknown({out_sum, out_status, out_invalid, out_meta, out_last})))
                else $error("fp32_reduction_tree no_unknown_result_when_valid failed");
            assert (!(in_valid && in_ready && (in_lane_mask === '0)))
                else $error("fp32_reduction_tree lane_mask_legal failed: zero active lanes");
            assert (!(add_input_fire && add_inflight_q))
                else $error("fp32_reduction_tree no_add_launch_while_inflight failed");
            assert (!(add_out_valid && add_out_ready && !add_inflight_q))
                else $error("fp32_reduction_tree no_result_without_matching_inflight_operation failed");
            if (add_out_valid && add_out_ready && add_inflight_q) begin
                assert (pair_q == add_launch_pair_q)
                    else $error("fp32_reduction_tree reduction_result_valid_has_matching_pair_id failed");
                assert (width_q == add_launch_width_q)
                    else $error("fp32_reduction_tree reduction_result_valid_has_matching_width failed");
                assert (add_in_a == add_launch_a_q && add_in_b == add_launch_b_q)
                    else $error("fp32_reduction_tree add_operands_stable_until_result failed");
            end

            if ($past(rst_n) && $past(out_valid && !out_ready)) begin
                assert (out_valid)
                    else $error("fp32_reduction_tree valid_stable_until_ready failed");
                assert ($stable(out_sum))
                    else $error("fp32_reduction_tree payload_stable_until_ready failed");
                assert ($stable(out_meta))
                    else $error("fp32_reduction_tree metadata_stable_until_ready failed");
                assert ($stable(out_last))
                    else $error("fp32_reduction_tree last_stable_until_ready failed");
            end
        end
    end
`endif
endmodule

`default_nettype wire
