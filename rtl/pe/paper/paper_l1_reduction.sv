`default_nettype none

module paper_l1_reduction #(
    parameter int META_W = 16,
    parameter bit ASSERT_ON_INVALID = 1'b1
) (
    input  logic              clk,
    input  logic              rst_n,

    input  logic              in_valid,
    output logic              in_ready,
    input  logic [8*32-1:0]   in_values_fp32,
    input  logic [7:0]        in_mask,
    input  logic [META_W-1:0] in_meta,
    input  logic              in_last,

    output logic              out_valid,
    input  logic              out_ready,
    output logic [31:0]       out_sum_fp32,
    output logic [7:0]        out_status,
    output logic              out_invalid,
    output logic [META_W-1:0] out_meta,
    output logic              out_last
);
    typedef enum logic [1:0] {
        ST_IDLE,
        ST_ADD_SEND,
        ST_ADD_WAIT,
        ST_OUTPUT
    } state_e;

    state_e state_q;
    logic [2:0] step_q;
    logic [31:0] values_q [0:7];
    logic [31:0] pair01_q;
    logic [31:0] pair23_q;
    logic [31:0] pair45_q;
    logic [31:0] pair67_q;
    logic [31:0] half0_q;
    logic [31:0] half1_q;
    logic [7:0] status_q;
    logic invalid_q;
    logic [META_W-1:0] meta_q;
    logic last_q;

    logic add_in_valid;
    logic add_in_ready;
    logic [31:0] add_in_a;
    logic [31:0] add_in_b;
    logic add_out_valid;
    logic add_out_ready;
    logic [31:0] add_out_result;
    logic [7:0] add_out_status;
    logic add_out_invalid;

    logic out_valid_q;
    logic [31:0] out_sum_q;
    logic [7:0] out_status_q;
    logic out_invalid_q;
    logic [META_W-1:0] out_meta_q;
    logic out_last_q;

    wire input_fire = in_valid && in_ready;
    wire add_input_fire = add_in_valid && add_in_ready;
    wire add_output_fire = add_out_valid && add_out_ready;
    wire output_fire = out_valid && out_ready;

    assign in_ready = (state_q == ST_IDLE) && !out_valid_q;
    assign out_valid = out_valid_q;
    assign out_sum_fp32 = out_sum_q;
    assign out_status = out_status_q;
    assign out_invalid = out_invalid_q;
    assign out_meta = out_meta_q;
    assign out_last = out_last_q;
    assign add_in_valid = (state_q == ST_ADD_SEND);
    assign add_out_ready = (state_q == ST_ADD_WAIT);

    always_comb begin
        add_in_a = 32'd0;
        add_in_b = 32'd0;
        unique case (step_q)
            3'd0: begin add_in_a = values_q[0]; add_in_b = values_q[1]; end
            3'd1: begin add_in_a = values_q[2]; add_in_b = values_q[3]; end
            3'd2: begin add_in_a = values_q[4]; add_in_b = values_q[5]; end
            3'd3: begin add_in_a = values_q[6]; add_in_b = values_q[7]; end
            3'd4: begin add_in_a = pair01_q; add_in_b = pair23_q; end
            3'd5: begin add_in_a = pair45_q; add_in_b = pair67_q; end
            default: begin add_in_a = half0_q; add_in_b = half1_q; end
        endcase
    end

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
        .out_meta    (),
        .out_last    ()
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            step_q <= 3'd0;
            for (int i = 0; i < 8; i++) begin
                values_q[i] <= 32'd0;
            end
            pair01_q <= 32'd0;
            pair23_q <= 32'd0;
            pair45_q <= 32'd0;
            pair67_q <= 32'd0;
            half0_q <= 32'd0;
            half1_q <= 32'd0;
            status_q <= 8'd0;
            invalid_q <= 1'b0;
            meta_q <= '0;
            last_q <= 1'b0;
            out_valid_q <= 1'b0;
            out_sum_q <= 32'd0;
            out_status_q <= 8'd0;
            out_invalid_q <= 1'b0;
            out_meta_q <= '0;
            out_last_q <= 1'b0;
        end else begin
            if (output_fire) begin
                out_valid_q <= 1'b0;
            end

            unique case (state_q)
                ST_IDLE: begin
                    if (input_fire) begin
                        for (int i = 0; i < 8; i++) begin
                            values_q[i] <= in_mask[i] ? in_values_fp32[i*32 +: 32] : 32'd0;
                        end
                        pair01_q <= 32'd0;
                        pair23_q <= 32'd0;
                        pair45_q <= 32'd0;
                        pair67_q <= 32'd0;
                        half0_q <= 32'd0;
                        half1_q <= 32'd0;
                        status_q <= 8'd0;
                        invalid_q <= 1'b0;
                        meta_q <= in_meta;
                        last_q <= in_last;
                        step_q <= 3'd0;
                        state_q <= ST_ADD_SEND;
                    end
                end

                ST_ADD_SEND: begin
                    if (add_input_fire) begin
                        state_q <= ST_ADD_WAIT;
                    end
                end

                ST_ADD_WAIT: begin
                    if (add_output_fire) begin
                        status_q <= status_q | add_out_status;
                        invalid_q <= invalid_q | add_out_invalid;
                        unique case (step_q)
                            3'd0: pair01_q <= add_out_result;
                            3'd1: pair23_q <= add_out_result;
                            3'd2: pair45_q <= add_out_result;
                            3'd3: pair67_q <= add_out_result;
                            3'd4: half0_q <= add_out_result;
                            3'd5: half1_q <= add_out_result;
                            default: begin
                                out_sum_q <= add_out_result;
                                out_status_q <= status_q | add_out_status;
                                out_invalid_q <= invalid_q | add_out_invalid;
                                out_meta_q <= meta_q;
                                out_last_q <= last_q;
                                out_valid_q <= 1'b1;
                            end
                        endcase
                        if (step_q == 3'd6) begin
                            state_q <= ST_OUTPUT;
                        end else begin
                            step_q <= step_q + 3'd1;
                            state_q <= ST_ADD_SEND;
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
            assert (!(in_valid && in_ready && $isunknown(in_mask)))
                else $error("paper_l1_reduction l1_reduction_order_legal mask unknown failed");
            if ($past(rst_n) && $past(out_valid && !out_ready)) begin
                assert (out_valid)
                    else $error("paper_l1_reduction output_stable_until_ready valid failed");
                assert ($stable({out_sum_fp32, out_status, out_invalid, out_meta, out_last}))
                    else $error("paper_l1_reduction output_stable_until_ready payload failed");
            end
        end
    end
`endif
endmodule

`default_nettype wire
