`default_nettype none

module softmax_normalization #(
    parameter int META_W = 1,
    parameter int TOKEN_W = 5,
    parameter bit ASSERT_ON_INVALID = 1'b1
) (
    input  logic                clk,
    input  logic                rst_n,
    input  logic                clear,

    input  logic                start_valid,
    output logic                start_ready,
    input  logic [31:0]         start_max,
    input  logic [31:0]         start_exp_sum,
    input  logic [META_W-1:0]   start_meta,

    input  logic                score_valid,
    output logic                score_ready,
    input  logic [31:0]         score_value,
    input  logic [TOKEN_W-1:0]  score_index,
    input  logic                score_last,

    output logic                prob_valid,
    input  logic                prob_ready,
    output logic [31:0]         prob_value,
    output logic [TOKEN_W-1:0]  prob_index,
    output logic                prob_last,
    output logic [7:0]          prob_status,
    output logic                prob_invalid,
    output logic [META_W-1:0]   prob_meta,
    output logic                busy
);
    localparam logic [31:0] FP32_ZERO = 32'h0000_0000;

    typedef enum logic [3:0] {
        ST_IDLE,
        ST_RECIP_SEND,
        ST_RECIP_WAIT,
        ST_WAIT_SCORE,
        ST_DELTA_SEND,
        ST_DELTA_WAIT,
        ST_EXP_SEND,
        ST_EXP_WAIT,
        ST_PROB_SEND,
        ST_PROB_WAIT,
        ST_OUTPUT
    } state_e;

    state_e state_q;
    logic [31:0] max_q;
    logic [31:0] exp_sum_q;
    logic [31:0] inv_sum_q;
    logic [31:0] score_q;
    logic [31:0] delta_q;
    logic [31:0] numerator_q;
    logic [TOKEN_W-1:0] score_index_q;
    logic score_last_q;
    logic [META_W-1:0] meta_q;
    logic [7:0] status_q;
    logic invalid_q;
    logic prob_valid_q;
    logic [31:0] prob_value_q;
    logic [TOKEN_W-1:0] prob_index_q;
    logic prob_last_q;
    logic [7:0] prob_status_q;
    logic prob_invalid_q;

    logic recip_in_valid;
    logic recip_in_ready;
    logic recip_out_valid;
    logic recip_out_ready;
    logic [31:0] recip_out_result;
    logic [7:0] recip_out_status;
    logic recip_out_invalid;

    logic add_in_valid;
    logic add_in_ready;
    logic add_out_valid;
    logic add_out_ready;
    logic [31:0] add_out_result;
    logic [7:0] add_out_status;
    logic add_out_invalid;

    logic exp_in_valid;
    logic exp_in_ready;
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

    wire start_fire = start_valid && start_ready;
    wire score_fire = score_valid && score_ready;
    wire prob_fire = prob_valid && prob_ready;
    wire recip_input_fire = recip_in_valid && recip_in_ready;
    wire recip_output_fire = recip_out_valid && recip_out_ready;
    wire add_input_fire = add_in_valid && add_in_ready;
    wire add_output_fire = add_out_valid && add_out_ready;
    wire exp_input_fire = exp_in_valid && exp_in_ready;
    wire exp_output_fire = exp_out_valid && exp_out_ready;
    wire mac_input_fire = mac_in_valid && mac_in_ready;
    wire mac_output_fire = mac_out_valid && mac_out_ready;

    initial begin
        if (META_W <= 0) begin
            $fatal(1, "softmax_normalization META_W must be positive");
        end
        if (TOKEN_W <= 0) begin
            $fatal(1, "softmax_normalization TOKEN_W must be positive");
        end
    end

    function automatic logic [31:0] fp32_neg(input logic [31:0] value);
        fp32_neg = {~value[31], value[30:0]};
    endfunction

    assign start_ready = (state_q == ST_IDLE);
    assign score_ready = (state_q == ST_WAIT_SCORE) && !prob_valid_q;
    assign prob_valid = prob_valid_q;
    assign prob_value = prob_value_q;
    assign prob_index = prob_index_q;
    assign prob_last = prob_last_q;
    assign prob_status = prob_status_q;
    assign prob_invalid = prob_invalid_q;
    assign prob_meta = meta_q;
    assign busy = (state_q != ST_IDLE) || prob_valid_q;

    assign recip_in_valid = (state_q == ST_RECIP_SEND);
    assign recip_out_ready = (state_q == ST_RECIP_WAIT);
    assign add_in_valid = (state_q == ST_DELTA_SEND);
    assign add_out_ready = (state_q == ST_DELTA_WAIT);
    assign exp_in_valid = (state_q == ST_EXP_SEND);
    assign exp_out_ready = (state_q == ST_EXP_WAIT);
    assign mac_in_valid = (state_q == ST_PROB_SEND);
    assign mac_out_ready = (state_q == ST_PROB_WAIT);

    fp32_recip_wrapper #(
        .META_W(META_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_recip (
        .clk         (clk),
        .rst_n       (rst_n),
        .in_valid    (recip_in_valid),
        .in_ready    (recip_in_ready),
        .in_a        (exp_sum_q),
        .in_meta     (meta_q),
        .in_last     (1'b0),
        .out_valid   (recip_out_valid),
        .out_ready   (recip_out_ready),
        .out_result  (recip_out_result),
        .out_status  (recip_out_status),
        .out_invalid (recip_out_invalid),
        .out_meta    (),
        .out_last    ()
    );

    fp32_add_wrapper #(
        .META_W(META_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_delta_add (
        .clk         (clk),
        .rst_n       (rst_n),
        .in_valid    (add_in_valid),
        .in_ready    (add_in_ready),
        .in_a        (score_q),
        .in_b        (fp32_neg(max_q)),
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
        .in_a        (delta_q),
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
    ) u_prob_mul (
        .clk         (clk),
        .rst_n       (rst_n),
        .in_valid    (mac_in_valid),
        .in_ready    (mac_in_ready),
        .in_a        (numerator_q),
        .in_b        (inv_sum_q),
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
            state_q <= ST_IDLE;
            max_q <= 32'd0;
            exp_sum_q <= 32'd0;
            inv_sum_q <= 32'd0;
            score_q <= 32'd0;
            delta_q <= 32'd0;
            numerator_q <= 32'd0;
            score_index_q <= '0;
            score_last_q <= 1'b0;
            meta_q <= '0;
            status_q <= 8'd0;
            invalid_q <= 1'b0;
            prob_valid_q <= 1'b0;
            prob_value_q <= 32'd0;
            prob_index_q <= '0;
            prob_last_q <= 1'b0;
            prob_status_q <= 8'd0;
            prob_invalid_q <= 1'b0;
        end else begin
            if (clear) begin
                state_q <= ST_IDLE;
                max_q <= 32'd0;
                exp_sum_q <= 32'd0;
                inv_sum_q <= 32'd0;
                score_q <= 32'd0;
                delta_q <= 32'd0;
                numerator_q <= 32'd0;
                score_index_q <= '0;
                score_last_q <= 1'b0;
                meta_q <= '0;
                status_q <= 8'd0;
                invalid_q <= 1'b0;
                prob_valid_q <= 1'b0;
                prob_value_q <= 32'd0;
                prob_index_q <= '0;
                prob_last_q <= 1'b0;
                prob_status_q <= 8'd0;
                prob_invalid_q <= 1'b0;
            end else begin
                if (prob_fire) begin
                    prob_valid_q <= 1'b0;
                    if (prob_last_q) begin
                        state_q <= ST_IDLE;
                    end else begin
                        state_q <= ST_WAIT_SCORE;
                    end
                end

                unique case (state_q)
                    ST_IDLE: begin
                        if (start_fire) begin
                            max_q <= start_max;
                            exp_sum_q <= start_exp_sum;
                            meta_q <= start_meta;
                            status_q <= 8'd0;
                            invalid_q <= 1'b0;
                            state_q <= ST_RECIP_SEND;
                        end
                    end

                    ST_RECIP_SEND: begin
                        if (recip_input_fire) begin
                            state_q <= ST_RECIP_WAIT;
                        end
                    end

                    ST_RECIP_WAIT: begin
                        if (recip_output_fire) begin
                            inv_sum_q <= recip_out_result;
                            status_q <= status_q | recip_out_status;
                            invalid_q <= invalid_q | recip_out_invalid;
                            state_q <= ST_WAIT_SCORE;
                        end
                    end

                    ST_WAIT_SCORE: begin
                        if (score_fire) begin
                            score_q <= score_value;
                            score_index_q <= score_index;
                            score_last_q <= score_last;
                            state_q <= ST_DELTA_SEND;
                        end
                    end

                    ST_DELTA_SEND: begin
                        if (add_input_fire) begin
                            state_q <= ST_DELTA_WAIT;
                        end
                    end

                    ST_DELTA_WAIT: begin
                        if (add_output_fire) begin
                            delta_q <= add_out_result;
                            status_q <= status_q | add_out_status;
                            invalid_q <= invalid_q | add_out_invalid;
                            state_q <= ST_EXP_SEND;
                        end
                    end

                    ST_EXP_SEND: begin
                        if (exp_input_fire) begin
                            state_q <= ST_EXP_WAIT;
                        end
                    end

                    ST_EXP_WAIT: begin
                        if (exp_output_fire) begin
                            numerator_q <= exp_out_result;
                            status_q <= status_q | exp_out_status;
                            invalid_q <= invalid_q | exp_out_invalid;
                            state_q <= ST_PROB_SEND;
                        end
                    end

                    ST_PROB_SEND: begin
                        if (mac_input_fire) begin
                            state_q <= ST_PROB_WAIT;
                        end
                    end

                    ST_PROB_WAIT: begin
                        if (mac_output_fire) begin
                            prob_value_q <= mac_out_result;
                            prob_index_q <= score_index_q;
                            prob_last_q <= score_last_q;
                            prob_status_q <= status_q | mac_out_status;
                            prob_invalid_q <= invalid_q | mac_out_invalid;
                            status_q <= status_q | mac_out_status;
                            invalid_q <= invalid_q | mac_out_invalid;
                            prob_valid_q <= 1'b1;
                            state_q <= ST_OUTPUT;
                        end
                    end

                    ST_OUTPUT: begin
                        if (prob_fire) begin
                            if (prob_last_q) begin
                                state_q <= ST_IDLE;
                            end else begin
                                state_q <= ST_WAIT_SCORE;
                            end
                        end
                    end

                    default: state_q <= ST_IDLE;
                endcase
            end
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n && !clear) begin
            assert (!(prob_valid && $isunknown({prob_value, prob_index, prob_last, prob_status, prob_invalid, prob_meta})))
                else $error("softmax_normalization no_unknown_output_when_valid failed");
            if ($past(rst_n) && $past(score_valid && !score_ready)) begin
                assert (score_valid)
                    else $error("softmax_normalization valid_stable_until_ready failed");
                assert ($stable({score_value, score_index, score_last}))
                    else $error("softmax_normalization payload_stable_until_ready failed");
            end
            if ($past(rst_n) && $past(prob_valid && !prob_ready)) begin
                assert (prob_valid)
                    else $error("softmax_normalization output valid dropped under backpressure");
                assert ($stable({prob_value, prob_index, prob_last, prob_status, prob_invalid, prob_meta}))
                    else $error("softmax_normalization output stable until ready failed");
            end
        end
    end
`endif
endmodule

`default_nettype wire

