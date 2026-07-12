`default_nettype none

module softmax_reduction #(
    parameter int META_W = 1,
    parameter bit ASSERT_ON_INVALID = 1'b1
) (
    input  logic              clk,
    input  logic              rst_n,
    input  logic              clear,

    input  logic              in_valid,
    output logic              in_ready,
    input  logic [31:0]       in_score,
    input  logic [META_W-1:0] in_meta,
    input  logic              in_last,

    output logic              final_valid,
    input  logic              final_ready,
    output logic [31:0]       final_max,
    output logic [31:0]       final_exp_sum,
    output logic [7:0]        final_status,
    output logic              final_invalid,
    output logic [META_W-1:0] final_meta,
    output logic              busy,
    output logic [31:0]       processed_count
);
    localparam logic [31:0] FP32_ONE = 32'h3F80_0000;
    localparam logic [31:0] FP32_ZERO = 32'h0000_0000;

    typedef enum logic [3:0] {
        ST_WAIT_SCORE,
        ST_DELTA_OLD_SEND,
        ST_DELTA_OLD_WAIT,
        ST_EXP_OLD_SEND,
        ST_EXP_OLD_WAIT,
        ST_TERM_SEND,
        ST_TERM_WAIT,
        ST_DELTA_X_SEND,
        ST_DELTA_X_WAIT,
        ST_EXP_X_SEND,
        ST_EXP_X_WAIT,
        ST_Z_SEND,
        ST_Z_WAIT,
        ST_FINAL
    } state_e;

    state_e state_q;
    logic [31:0] max_q;
    logic [31:0] exp_sum_q;
    logic [31:0] score_q;
    logic [31:0] new_max_q;
    logic [31:0] old_delta_q;
    logic [31:0] x_delta_q;
    logic [31:0] old_exp_q;
    logic [31:0] x_exp_q;
    logic [31:0] scaled_old_sum_q;
    logic score_last_q;
    logic [META_W-1:0] meta_q;
    logic [7:0] status_q;
    logic invalid_q;
    logic have_value_q;
    logic final_valid_q;

    logic add_in_valid;
    logic add_in_ready;
    logic [31:0] add_in_a;
    logic [31:0] add_in_b;
    logic add_out_valid;
    logic add_out_ready;
    logic [31:0] add_out_result;
    logic [7:0] add_out_status;
    logic add_out_invalid;

    logic exp_in_valid;
    logic exp_in_ready;
    logic [31:0] exp_in_a;
    logic exp_out_valid;
    logic exp_out_ready;
    logic [31:0] exp_out_result;
    logic [7:0] exp_out_status;
    logic exp_out_invalid;

    logic mac_in_valid;
    logic mac_in_ready;
    logic mac_out_valid;
    logic mac_out_ready;
    logic [31:0] mac_out_result;
    logic [7:0] mac_out_status;
    logic mac_out_invalid;

    wire input_fire = in_valid && in_ready;
    wire final_fire = final_valid && final_ready;
    wire add_input_fire = add_in_valid && add_in_ready;
    wire add_output_fire = add_out_valid && add_out_ready;
    wire exp_input_fire = exp_in_valid && exp_in_ready;
    wire exp_output_fire = exp_out_valid && exp_out_ready;
    wire mac_input_fire = mac_in_valid && mac_in_ready;
    wire mac_output_fire = mac_out_valid && mac_out_ready;

    initial begin
        if (META_W <= 0) begin
            $fatal(1, "softmax_reduction META_W must be positive");
        end
    end

    function automatic logic [31:0] fp32_neg(input logic [31:0] value);
        fp32_neg = {~value[31], value[30:0]};
    endfunction

    function automatic logic fp32_gt(input logic [31:0] a, input logic [31:0] b);
        logic a_sign;
        logic b_sign;
        begin
            if ((a[30:0] == 31'd0) && (b[30:0] == 31'd0)) begin
                fp32_gt = 1'b0;
            end else begin
                a_sign = a[31];
                b_sign = b[31];
                if (a_sign != b_sign) begin
                    fp32_gt = b_sign;
                end else if (!a_sign) begin
                    fp32_gt = (a[30:0] > b[30:0]);
                end else begin
                    fp32_gt = (a[30:0] < b[30:0]);
                end
            end
        end
    endfunction

    assign in_ready = (state_q == ST_WAIT_SCORE) && !final_valid_q;
    assign final_valid = final_valid_q;
    assign final_max = max_q;
    assign final_exp_sum = exp_sum_q;
    assign final_status = status_q;
    assign final_invalid = invalid_q;
    assign final_meta = meta_q;
    assign busy = (state_q != ST_WAIT_SCORE) || final_valid_q;

    always_comb begin
        add_in_valid = 1'b0;
        add_in_a = 32'd0;
        add_in_b = 32'd0;
        unique case (state_q)
            ST_DELTA_OLD_SEND: begin
                add_in_valid = 1'b1;
                add_in_a = max_q;
                add_in_b = fp32_neg(new_max_q);
            end
            ST_DELTA_X_SEND: begin
                add_in_valid = 1'b1;
                add_in_a = score_q;
                add_in_b = fp32_neg(new_max_q);
            end
            ST_Z_SEND: begin
                add_in_valid = 1'b1;
                add_in_a = scaled_old_sum_q;
                add_in_b = x_exp_q;
            end
            default: begin
                add_in_valid = 1'b0;
                add_in_a = 32'd0;
                add_in_b = 32'd0;
            end
        endcase
    end

    assign add_out_ready =
        (state_q == ST_DELTA_OLD_WAIT) ||
        (state_q == ST_DELTA_X_WAIT) ||
        (state_q == ST_Z_WAIT);

    always_comb begin
        exp_in_valid = 1'b0;
        exp_in_a = 32'd0;
        unique case (state_q)
            ST_EXP_OLD_SEND: begin
                exp_in_valid = 1'b1;
                exp_in_a = old_delta_q;
            end
            ST_EXP_X_SEND: begin
                exp_in_valid = 1'b1;
                exp_in_a = x_delta_q;
            end
            default: begin
                exp_in_valid = 1'b0;
                exp_in_a = 32'd0;
            end
        endcase
    end

    assign exp_out_ready =
        (state_q == ST_EXP_OLD_WAIT) ||
        (state_q == ST_EXP_X_WAIT);

    assign mac_in_valid = (state_q == ST_TERM_SEND);
    assign mac_out_ready = (state_q == ST_TERM_WAIT);

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
        .in_last     (score_last_q),
        .out_valid   (add_out_valid),
        .out_ready   (add_out_ready),
        .out_result  (add_out_result),
        .out_status  (add_out_status),
        .out_invalid (add_out_invalid),
        .out_meta    (),
        .out_last    ()
    );

    fp32_exp_wrapper #(
        .META_W(META_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_exp (
        .clk         (clk),
        .rst_n       (rst_n),
        .in_valid    (exp_in_valid),
        .in_ready    (exp_in_ready),
        .in_a        (exp_in_a),
        .in_meta     (meta_q),
        .in_last     (score_last_q),
        .out_valid   (exp_out_valid),
        .out_ready   (exp_out_ready),
        .out_result  (exp_out_result),
        .out_status  (exp_out_status),
        .out_invalid (exp_out_invalid),
        .out_meta    (),
        .out_last    ()
    );

    fp32_mac_wrapper #(
        .META_W(META_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_scale_old_sum (
        .clk         (clk),
        .rst_n       (rst_n),
        .in_valid    (mac_in_valid),
        .in_ready    (mac_in_ready),
        .in_a        (exp_sum_q),
        .in_b        (old_exp_q),
        .in_c        (FP32_ZERO),
        .in_meta     (meta_q),
        .in_last     (score_last_q),
        .out_valid   (mac_out_valid),
        .out_ready   (mac_out_ready),
        .out_result  (mac_out_result),
        .out_status  (mac_out_status),
        .out_invalid (mac_out_invalid),
        .out_meta    (),
        .out_last    ()
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= ST_WAIT_SCORE;
            max_q <= 32'd0;
            exp_sum_q <= 32'd0;
            score_q <= 32'd0;
            new_max_q <= 32'd0;
            old_delta_q <= 32'd0;
            x_delta_q <= 32'd0;
            old_exp_q <= 32'd0;
            x_exp_q <= 32'd0;
            scaled_old_sum_q <= 32'd0;
            score_last_q <= 1'b0;
            meta_q <= '0;
            status_q <= 8'd0;
            invalid_q <= 1'b0;
            have_value_q <= 1'b0;
            final_valid_q <= 1'b0;
            processed_count <= 32'd0;
        end else begin
            if (clear) begin
                state_q <= ST_WAIT_SCORE;
                max_q <= 32'd0;
                exp_sum_q <= 32'd0;
                score_q <= 32'd0;
                new_max_q <= 32'd0;
                old_delta_q <= 32'd0;
                x_delta_q <= 32'd0;
                old_exp_q <= 32'd0;
                x_exp_q <= 32'd0;
                scaled_old_sum_q <= 32'd0;
                score_last_q <= 1'b0;
                meta_q <= '0;
                status_q <= 8'd0;
                invalid_q <= 1'b0;
                have_value_q <= 1'b0;
                final_valid_q <= 1'b0;
                processed_count <= 32'd0;
            end else begin
                if (final_fire) begin
                    final_valid_q <= 1'b0;
                    state_q <= ST_WAIT_SCORE;
                end

                unique case (state_q)
                    ST_WAIT_SCORE: begin
                        if (input_fire) begin
                            score_q <= in_score;
                            score_last_q <= in_last;
                            meta_q <= in_meta;
                            processed_count <= processed_count + 32'd1;
                            if (!have_value_q) begin
                                max_q <= in_score;
                                exp_sum_q <= FP32_ONE;
                                have_value_q <= 1'b1;
                                if (in_last) begin
                                    final_valid_q <= 1'b1;
                                    state_q <= ST_FINAL;
                                end
                            end else begin
                                new_max_q <= fp32_gt(in_score, max_q) ? in_score : max_q;
                                state_q <= ST_DELTA_OLD_SEND;
                            end
                        end
                    end

                    ST_DELTA_OLD_SEND: begin
                        if (add_input_fire) begin
                            state_q <= ST_DELTA_OLD_WAIT;
                        end
                    end

                    ST_DELTA_OLD_WAIT: begin
                        if (add_output_fire) begin
                            old_delta_q <= add_out_result;
                            status_q <= status_q | add_out_status;
                            invalid_q <= invalid_q | add_out_invalid;
                            state_q <= ST_EXP_OLD_SEND;
                        end
                    end

                    ST_EXP_OLD_SEND: begin
                        if (exp_input_fire) begin
                            state_q <= ST_EXP_OLD_WAIT;
                        end
                    end

                    ST_EXP_OLD_WAIT: begin
                        if (exp_output_fire) begin
                            old_exp_q <= exp_out_result;
                            status_q <= status_q | exp_out_status;
                            invalid_q <= invalid_q | exp_out_invalid;
                            state_q <= ST_TERM_SEND;
                        end
                    end

                    ST_TERM_SEND: begin
                        if (mac_input_fire) begin
                            state_q <= ST_TERM_WAIT;
                        end
                    end

                    ST_TERM_WAIT: begin
                        if (mac_output_fire) begin
                            scaled_old_sum_q <= mac_out_result;
                            status_q <= status_q | mac_out_status;
                            invalid_q <= invalid_q | mac_out_invalid;
                            state_q <= ST_DELTA_X_SEND;
                        end
                    end

                    ST_DELTA_X_SEND: begin
                        if (add_input_fire) begin
                            state_q <= ST_DELTA_X_WAIT;
                        end
                    end

                    ST_DELTA_X_WAIT: begin
                        if (add_output_fire) begin
                            x_delta_q <= add_out_result;
                            status_q <= status_q | add_out_status;
                            invalid_q <= invalid_q | add_out_invalid;
                            state_q <= ST_EXP_X_SEND;
                        end
                    end

                    ST_EXP_X_SEND: begin
                        if (exp_input_fire) begin
                            state_q <= ST_EXP_X_WAIT;
                        end
                    end

                    ST_EXP_X_WAIT: begin
                        if (exp_output_fire) begin
                            x_exp_q <= exp_out_result;
                            status_q <= status_q | exp_out_status;
                            invalid_q <= invalid_q | exp_out_invalid;
                            state_q <= ST_Z_SEND;
                        end
                    end

                    ST_Z_SEND: begin
                        if (add_input_fire) begin
                            state_q <= ST_Z_WAIT;
                        end
                    end

                    ST_Z_WAIT: begin
                        if (add_output_fire) begin
                            max_q <= new_max_q;
                            exp_sum_q <= add_out_result;
                            status_q <= status_q | add_out_status;
                            invalid_q <= invalid_q | add_out_invalid;
                            if (score_last_q) begin
                                final_valid_q <= 1'b1;
                                state_q <= ST_FINAL;
                            end else begin
                                state_q <= ST_WAIT_SCORE;
                            end
                        end
                    end

                    ST_FINAL: begin
                        if (final_fire) begin
                            state_q <= ST_WAIT_SCORE;
                        end
                    end

                    default: state_q <= ST_WAIT_SCORE;
                endcase
            end
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n && !clear) begin
            assert (!(final_valid && $isunknown({final_max, final_exp_sum, final_status, final_invalid, final_meta})))
                else $error("softmax_reduction no_unknown_output_when_valid failed");
            if ($past(rst_n) && $past(final_valid && !final_ready)) begin
                assert (final_valid)
                    else $error("softmax_reduction output valid dropped under backpressure");
                assert ($stable({final_max, final_exp_sum, final_status, final_invalid, final_meta}))
                    else $error("softmax_reduction output stable until ready failed");
            end
            if ($past(rst_n) && $past(in_valid && !in_ready)) begin
                assert (in_valid)
                    else $error("softmax_reduction valid_stable_until_ready failed");
                assert ($stable({in_score, in_meta, in_last}))
                    else $error("softmax_reduction payload_stable_until_ready failed");
            end
        end
    end
`endif
endmodule

`default_nettype wire

