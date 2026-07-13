`default_nettype none

module paper_pe_cell #(
    parameter int GROUP_INDEX = 0,
    parameter int ROW_INDEX = 0,
    parameter int COLUMN_INDEX = 0,
    parameter int PE_TYPE = 0, // 0: Type-A, 1: Type-B
    parameter int META_W = 16,
    parameter bit ASSERT_ON_INVALID = 1'b1
) (
    input  logic              clk,
    input  logic              rst_n,

    input  logic              op_valid,
    output logic              op_ready,
    input  logic [1:0]        op_mode,
    input  logic              op_active,
    input  logic              op_clear_acc,
    input  logic [15:0]       op_operand_a_fp16,
    input  logic [15:0]       op_operand_b_fp16,
    input  logic [31:0]       op_scalar_fp32,
    input  logic [31:0]       op_partial_sum_in,
    input  logic [META_W-1:0] op_meta,
    input  logic              op_last,

    output logic              out_valid,
    input  logic              out_ready,
    output logic [31:0]       out_product_fp32,
    output logic [31:0]       out_accumulator_fp32,
    output logic [31:0]       out_forwarded_partial_fp32,
    output logic [7:0]        out_status,
    output logic              out_invalid,
    output logic              out_active,
    output logic [META_W-1:0] out_meta,
    output logic              out_last
);
    localparam logic [1:0] MODE_INNER_PRODUCT = 2'd1;
    localparam logic [1:0] MODE_OUTER_PRODUCT = 2'd2;

    typedef enum logic [2:0] {
        ST_IDLE,
        ST_CONV_SEND,
        ST_CONV_WAIT,
        ST_MAC_WAIT,
        ST_OUTPUT
    } state_e;

    state_e state_q;

    logic [1:0] mode_q;
    logic active_q;
    logic clear_acc_q;
    logic [15:0] operand_a_q;
    logic [15:0] operand_b_q;
    logic [31:0] scalar_q;
    logic [31:0] partial_sum_q;
    logic [META_W-1:0] meta_q;
    logic last_q;
    logic [31:0] accumulator_q;

    logic conv_in_valid;
    logic conv_a_in_ready;
    logic conv_b_in_ready;
    logic conv_a_out_valid;
    logic conv_b_out_valid;
    logic conv_out_ready;
    logic [31:0] conv_a_data;
    logic [31:0] conv_b_data;
    logic conv_a_invalid;
    logic conv_b_invalid;
    logic conv_a_underflow;
    logic conv_b_underflow;

    logic mac_in_valid;
    logic mac_in_ready;
    logic mac_out_valid;
    logic mac_out_ready;
    logic [31:0] mac_a;
    logic [31:0] mac_b;
    logic [31:0] mac_c;
    logic [31:0] mac_result;
    logic [7:0] mac_status;
    logic mac_invalid;

    logic out_valid_q;
    logic [31:0] out_product_q;
    logic [31:0] out_accumulator_q;
    logic [31:0] out_forwarded_q;
    logic [7:0] out_status_q;
    logic out_invalid_q;
    logic out_active_q;
    logic [META_W-1:0] out_meta_q;
    logic out_last_q;

    wire input_fire = op_valid && op_ready;
    wire output_fire = out_valid && out_ready;
    wire conv_input_fire = conv_in_valid && conv_a_in_ready && conv_b_in_ready;
    wire conv_output_fire = conv_a_out_valid && conv_b_out_valid && conv_out_ready;
    wire mac_input_fire = mac_in_valid && mac_in_ready;
    wire mac_output_fire = mac_out_valid && mac_out_ready;

    initial begin
        if (GROUP_INDEX < 0 || GROUP_INDEX > 1) begin
            $fatal(1, "paper_pe_cell group index out of range");
        end
        if (ROW_INDEX < 0 || ROW_INDEX > 7) begin
            $fatal(1, "paper_pe_cell row index out of range");
        end
        if (COLUMN_INDEX < 0 || COLUMN_INDEX > 7) begin
            $fatal(1, "paper_pe_cell column index out of range");
        end
        if (!((PE_TYPE == 0 && ((COLUMN_INDEX % 2) == 0)) ||
              (PE_TYPE == 1 && ((COLUMN_INDEX % 2) == 1)))) begin
            $fatal(1, "paper_pe_cell pe_type_mapping_matches_spec failed");
        end
        if (META_W <= 0) begin
            $fatal(1, "paper_pe_cell META_W must be positive");
        end
    end

    assign op_ready = (state_q == ST_IDLE) && !out_valid_q;
    assign out_valid = out_valid_q;
    assign out_product_fp32 = out_product_q;
    assign out_accumulator_fp32 = out_accumulator_q;
    assign out_forwarded_partial_fp32 = out_forwarded_q;
    assign out_status = out_status_q;
    assign out_invalid = out_invalid_q;
    assign out_active = out_active_q;
    assign out_meta = out_meta_q;
    assign out_last = out_last_q;

    assign conv_in_valid = (state_q == ST_CONV_SEND);
    assign conv_out_ready = (state_q == ST_CONV_WAIT) && mac_in_ready;
    assign mac_in_valid = (state_q == ST_CONV_WAIT) && conv_a_out_valid && conv_b_out_valid;
    assign mac_out_ready = (state_q == ST_MAC_WAIT) && (!out_valid_q || out_ready);

    assign mac_a = (mode_q == MODE_OUTER_PRODUCT) ? scalar_q : conv_a_data;
    assign mac_b = conv_b_data;
    assign mac_c = (mode_q == MODE_OUTER_PRODUCT) ? (clear_acc_q ? 32'd0 : accumulator_q) : partial_sum_q;

    fp16_to_fp32 #(
        .META_W(META_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_conv_a (
        .clk                  (clk),
        .rst_n                (rst_n),
        .in_valid             (conv_in_valid),
        .in_ready             (conv_a_in_ready),
        .in_data              ((active_q && mode_q == MODE_INNER_PRODUCT) ? operand_a_q : 16'd0),
        .in_meta              (meta_q),
        .in_last              (last_q),
        .out_valid            (conv_a_out_valid),
        .out_ready            (conv_out_ready),
        .out_data             (conv_a_data),
        .out_meta             (),
        .out_last             (),
        .out_invalid          (conv_a_invalid),
        .out_underflow_or_ftz (conv_a_underflow)
    );

    fp16_to_fp32 #(
        .META_W(META_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_conv_b (
        .clk                  (clk),
        .rst_n                (rst_n),
        .in_valid             (conv_in_valid),
        .in_ready             (conv_b_in_ready),
        .in_data              (active_q ? operand_b_q : 16'd0),
        .in_meta              (meta_q),
        .in_last              (last_q),
        .out_valid            (conv_b_out_valid),
        .out_ready            (conv_out_ready),
        .out_data             (conv_b_data),
        .out_meta             (),
        .out_last             (),
        .out_invalid          (conv_b_invalid),
        .out_underflow_or_ftz (conv_b_underflow)
    );

    fp32_mac_wrapper #(
        .META_W(META_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_mac (
        .clk         (clk),
        .rst_n       (rst_n),
        .in_valid    (mac_in_valid),
        .in_ready    (mac_in_ready),
        .in_a        (mac_a),
        .in_b        (mac_b),
        .in_c        (mac_c),
        .in_meta     (meta_q),
        .in_last     (last_q),
        .out_valid   (mac_out_valid),
        .out_ready   (mac_out_ready),
        .out_result  (mac_result),
        .out_status  (mac_status),
        .out_invalid (mac_invalid),
        .out_meta    (),
        .out_last    ()
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            mode_q <= MODE_INNER_PRODUCT;
            active_q <= 1'b0;
            clear_acc_q <= 1'b0;
            operand_a_q <= 16'd0;
            operand_b_q <= 16'd0;
            scalar_q <= 32'd0;
            partial_sum_q <= 32'd0;
            meta_q <= '0;
            last_q <= 1'b0;
            accumulator_q <= 32'd0;
            out_valid_q <= 1'b0;
            out_product_q <= 32'd0;
            out_accumulator_q <= 32'd0;
            out_forwarded_q <= 32'd0;
            out_status_q <= 8'd0;
            out_invalid_q <= 1'b0;
            out_active_q <= 1'b0;
            out_meta_q <= '0;
            out_last_q <= 1'b0;
        end else begin
            if (output_fire) begin
                out_valid_q <= 1'b0;
            end

            unique case (state_q)
                ST_IDLE: begin
                    if (input_fire) begin
                        mode_q <= op_mode;
                        active_q <= op_active;
                        clear_acc_q <= op_clear_acc;
                        operand_a_q <= op_operand_a_fp16;
                        operand_b_q <= op_operand_b_fp16;
                        scalar_q <= op_scalar_fp32;
                        partial_sum_q <= op_partial_sum_in;
                        meta_q <= op_meta;
                        last_q <= op_last;
                        if (op_clear_acc) begin
                            accumulator_q <= 32'd0;
                        end
                        state_q <= ST_CONV_SEND;
                    end
                end

                ST_CONV_SEND: begin
                    if (conv_input_fire) begin
                        state_q <= ST_CONV_WAIT;
                    end
                end

                ST_CONV_WAIT: begin
                    if (conv_output_fire && mac_input_fire) begin
                        state_q <= ST_MAC_WAIT;
                    end
                end

                ST_MAC_WAIT: begin
                    if (mac_output_fire) begin
                        if (mode_q == MODE_OUTER_PRODUCT) begin
                            accumulator_q <= mac_result;
                            out_product_q <= 32'd0;
                            out_accumulator_q <= mac_result;
                            out_forwarded_q <= mac_result;
                        end else begin
                            out_product_q <= mac_result;
                            out_accumulator_q <= accumulator_q;
                            out_forwarded_q <= mac_result;
                        end
                        out_status_q <= mac_status;
                        out_invalid_q <= mac_invalid | conv_a_invalid | conv_b_invalid;
                        out_active_q <= active_q;
                        out_meta_q <= meta_q;
                        out_last_q <= last_q;
                        out_valid_q <= 1'b1;
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
            assert ((op_mode == MODE_INNER_PRODUCT) || (op_mode == MODE_OUTER_PRODUCT))
                else $error("paper_pe_cell unsupported mode");
            if ($past(rst_n) && $past(op_valid && !op_ready)) begin
                assert (op_valid)
                    else $error("paper_pe_cell command_stable_until_ready failed");
                assert ($stable({op_mode, op_active, op_clear_acc, op_operand_a_fp16,
                                 op_operand_b_fp16, op_scalar_fp32, op_partial_sum_in,
                                 op_meta, op_last}))
                    else $error("paper_pe_cell payload_stable_until_ready failed");
            end
            if ($past(rst_n) && $past(out_valid && !out_ready)) begin
                assert (out_valid)
                    else $error("paper_pe_cell output_stable_until_ready valid failed");
                assert ($stable({out_product_fp32, out_accumulator_fp32,
                                 out_forwarded_partial_fp32, out_status, out_invalid,
                                 out_active, out_meta, out_last}))
                    else $error("paper_pe_cell output_stable_until_ready payload failed");
            end
            assert (!(out_valid && $isunknown({out_product_fp32, out_accumulator_fp32,
                                               out_forwarded_partial_fp32, out_status,
                                               out_invalid, out_active, out_meta, out_last})))
                else $error("paper_pe_cell no_unknown_output_when_valid failed");
        end
    end
`endif
endmodule

`default_nettype wire
