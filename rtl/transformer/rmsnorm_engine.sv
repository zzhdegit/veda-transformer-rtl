`default_nettype none

module rmsnorm_engine #(
    parameter int D_MODEL = 16,
    parameter int META_W = 16,
    parameter int COUNTER_W = 64,
    parameter bit ASSERT_ON_INVALID = 1'b1,
    localparam int DIM_W = (D_MODEL <= 1) ? 1 : $clog2(D_MODEL)
) (
    input  logic                         clk,
    input  logic                         rst_n,

    input  logic                         clear,

    input  logic                         gamma_valid,
    output logic                         gamma_ready,
    input  logic [DIM_W-1:0]             gamma_dim,
    input  logic [15:0]                  gamma_data_fp16,
    input  logic                         gamma_commit,

    input  logic                         input_valid,
    output logic                         input_ready,
    input  logic [DIM_W-1:0]             input_dim,
    input  logic [31:0]                  input_data_fp32,
    input  logic                         input_last,
    input  logic [META_W-1:0]            input_meta,
    input  logic                         input_commit,

    input  logic                         start_valid,
    output logic                         start_ready,
    input  logic [META_W-1:0]            start_meta,

    output logic                         output_valid,
    input  logic                         output_ready,
    output logic [DIM_W-1:0]             output_dim,
    output logic [15:0]                  output_data_fp16,
    output logic [7:0]                   output_status,
    output logic                         output_invalid,
    output logic [META_W-1:0]            output_meta,
    output logic                         output_last,

    output logic                         done_valid,
    input  logic                         done_ready,
    output logic [7:0]                   done_status,
    output logic                         done_invalid,
    output logic [META_W-1:0]            done_meta,

    output logic [31:0]                  debug_sum_sq,
    output logic [31:0]                  debug_inv_rms,

    output logic [COUNTER_W-1:0]         perf_reduce_cycles,
    output logic [COUNTER_W-1:0]         perf_apply_cycles,
    output logic [COUNTER_W-1:0]         perf_sfu_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_output_stall_cycles
);
    localparam logic [31:0] FP32_ZERO = 32'h0000_0000;
    localparam logic [31:0] EPS_FP32 = 32'h3727_C5AC;
    localparam logic [7:0] STATUS_OK = 8'h00;
    localparam logic [7:0] STATUS_INCOMPLETE = 8'hB1;
    localparam logic [7:0] STATUS_RANGE = 8'hB2;

    typedef enum logic [4:0] {
        ST_IDLE,
        ST_REDUCE_SEND,
        ST_REDUCE_WAIT,
        ST_SCALE_SEND,
        ST_SCALE_WAIT,
        ST_ADD_EPS_SEND,
        ST_ADD_EPS_WAIT,
        ST_SQRT_SEND,
        ST_SQRT_WAIT,
        ST_RECIP_SEND,
        ST_RECIP_WAIT,
        ST_GAMMA_SEND,
        ST_GAMMA_WAIT,
        ST_MUL1_SEND,
        ST_MUL1_WAIT,
        ST_MUL2_SEND,
        ST_MUL2_WAIT,
        ST_QUANT_SEND,
        ST_QUANT_WAIT,
        ST_DONE
    } state_e;

    state_e state_q;

    logic [31:0] input_mem [0:D_MODEL-1];
    logic [15:0] gamma_mem [0:D_MODEL-1];
    logic [D_MODEL-1:0] input_loaded_mask_q;
    logic [D_MODEL-1:0] gamma_loaded_mask_q;
    logic input_complete_q;
    logic gamma_complete_q;
    logic load_error_q;
    logic [META_W-1:0] input_meta_q;

    logic [DIM_W-1:0] reduce_index_q;
    logic [DIM_W-1:0] apply_index_q;
    logic [META_W-1:0] meta_q;
    logic [7:0] status_q;
    logic invalid_q;
    logic [31:0] acc_q;
    logic [31:0] mean_sq_q;
    logic [31:0] den_q;
    logic [31:0] inv_rms_q;
    logic [31:0] gamma_fp32_q;
    logic [31:0] mul1_q;
    logic [31:0] norm_fp32_q;

    logic done_valid_q;
    logic [7:0] done_status_q;
    logic done_invalid_q;
    logic [META_W-1:0] done_meta_q;

    logic mac_in_valid;
    logic mac_in_ready;
    logic [31:0] mac_in_a;
    logic [31:0] mac_in_b;
    logic [31:0] mac_in_c;
    logic mac_out_valid;
    logic mac_out_ready;
    logic [31:0] mac_out_result;
    logic [7:0] mac_out_status;
    logic mac_out_invalid;

    logic add_in_valid;
    logic add_in_ready;
    logic add_out_valid;
    logic add_out_ready;
    logic [31:0] add_out_result;
    logic [7:0] add_out_status;
    logic add_out_invalid;

    logic sqrt_in_valid;
    logic sqrt_in_ready;
    logic sqrt_out_valid;
    logic sqrt_out_ready;
    logic [31:0] sqrt_out_result;
    logic [7:0] sqrt_out_status;
    logic sqrt_out_invalid;

    logic recip_in_valid;
    logic recip_in_ready;
    logic recip_out_valid;
    logic recip_out_ready;
    logic [31:0] recip_out_result;
    logic [7:0] recip_out_status;
    logic recip_out_invalid;

    logic gamma_conv_in_valid;
    logic gamma_conv_in_ready;
    logic gamma_conv_out_valid;
    logic gamma_conv_out_ready;
    logic [31:0] gamma_conv_out_data;
    logic gamma_conv_out_invalid;

    logic quant_in_valid;
    logic quant_in_ready;
    logic quant_out_valid;
    logic quant_out_ready;
    logic [15:0] quant_out_data;
    logic quant_out_invalid;
    logic quant_out_overflow;
    logic quant_out_underflow_or_ftz;
    logic quant_out_inexact;

    wire gamma_fire = gamma_valid && gamma_ready;
    wire input_fire = input_valid && input_ready;
    wire start_fire = start_valid && start_ready;
    wire done_fire = done_valid && done_ready;
    wire mac_out_fire = mac_out_valid && mac_out_ready;
    wire add_out_fire = add_out_valid && add_out_ready;
    wire sqrt_out_fire = sqrt_out_valid && sqrt_out_ready;
    wire recip_out_fire = recip_out_valid && recip_out_ready;
    wire gamma_conv_out_fire = gamma_conv_out_valid && gamma_conv_out_ready;
    wire quant_out_fire = quant_out_valid && quant_out_ready;
    wire reduce_last = reduce_index_q == DIM_W'(D_MODEL - 1);
    wire apply_last = apply_index_q == DIM_W'(D_MODEL - 1);
    wire input_range_legal = int'(input_dim) < D_MODEL;
    wire gamma_range_legal = int'(gamma_dim) < D_MODEL;

    function automatic logic [31:0] mean_scale_fp32;
        begin
            unique case (D_MODEL)
                8: mean_scale_fp32 = 32'h3E00_0000;
                16: mean_scale_fp32 = 32'h3D80_0000;
                32: mean_scale_fp32 = 32'h3D00_0000;
                64: mean_scale_fp32 = 32'h3C80_0000;
                128: mean_scale_fp32 = 32'h3C00_0000;
                default: mean_scale_fp32 = 32'h0000_0000;
            endcase
        end
    endfunction

    initial begin
        if (D_MODEL <= 0 || META_W <= 0 || COUNTER_W <= 0) begin
            $fatal(1, "rmsnorm_engine parameters must be positive");
        end
        if ((D_MODEL & (D_MODEL - 1)) != 0) begin
            $fatal(1, "rmsnorm_engine D_MODEL must be power of two");
        end
        if (!(D_MODEL == 8 || D_MODEL == 16 || D_MODEL == 32 ||
              D_MODEL == 64 || D_MODEL == 128)) begin
            $fatal(1, "rmsnorm_engine unsupported D_MODEL mean scale");
        end
    end

    assign gamma_ready = state_q == ST_IDLE;
    assign input_ready = state_q == ST_IDLE;
    assign start_ready = (state_q == ST_IDLE) && !done_valid_q;

    always_comb begin
        mac_in_valid = 1'b0;
        mac_in_a = FP32_ZERO;
        mac_in_b = FP32_ZERO;
        mac_in_c = FP32_ZERO;
        unique case (state_q)
            ST_REDUCE_SEND: begin
                mac_in_valid = 1'b1;
                mac_in_a = input_mem[int'(reduce_index_q)];
                mac_in_b = input_mem[int'(reduce_index_q)];
                mac_in_c = acc_q;
            end
            ST_SCALE_SEND: begin
                mac_in_valid = 1'b1;
                mac_in_a = acc_q;
                mac_in_b = mean_scale_fp32();
                mac_in_c = FP32_ZERO;
            end
            ST_MUL1_SEND: begin
                mac_in_valid = 1'b1;
                mac_in_a = input_mem[int'(apply_index_q)];
                mac_in_b = inv_rms_q;
                mac_in_c = FP32_ZERO;
            end
            ST_MUL2_SEND: begin
                mac_in_valid = 1'b1;
                mac_in_a = mul1_q;
                mac_in_b = gamma_fp32_q;
                mac_in_c = FP32_ZERO;
            end
            default: begin
                mac_in_valid = 1'b0;
            end
        endcase
    end

    assign mac_out_ready = state_q == ST_REDUCE_WAIT ||
                           state_q == ST_SCALE_WAIT ||
                           state_q == ST_MUL1_WAIT ||
                           state_q == ST_MUL2_WAIT;
    assign add_in_valid = state_q == ST_ADD_EPS_SEND;
    assign add_out_ready = state_q == ST_ADD_EPS_WAIT;
    assign sqrt_in_valid = state_q == ST_SQRT_SEND;
    assign sqrt_out_ready = state_q == ST_SQRT_WAIT;
    assign recip_in_valid = state_q == ST_RECIP_SEND;
    assign recip_out_ready = state_q == ST_RECIP_WAIT;
    assign gamma_conv_in_valid = state_q == ST_GAMMA_SEND;
    assign gamma_conv_out_ready = state_q == ST_GAMMA_WAIT;
    assign quant_in_valid = state_q == ST_QUANT_SEND;
    assign quant_out_ready = (state_q == ST_QUANT_WAIT) && output_ready;

    assign output_valid = (state_q == ST_QUANT_WAIT) && quant_out_valid;
    assign output_dim = apply_index_q;
    assign output_data_fp16 = quant_out_data;
    assign output_status = status_q;
    assign output_invalid = invalid_q | quant_out_invalid;
    assign output_meta = meta_q;
    assign output_last = apply_last;

    assign done_valid = done_valid_q;
    assign done_status = done_status_q;
    assign done_invalid = done_invalid_q;
    assign done_meta = done_meta_q;
    assign debug_sum_sq = acc_q;
    assign debug_inv_rms = inv_rms_q;

    fp32_mac_wrapper #(
        .META_W(META_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_mac (
        .clk         (clk),
        .rst_n       (rst_n),
        .in_valid    (mac_in_valid),
        .in_ready    (mac_in_ready),
        .in_a        (mac_in_a),
        .in_b        (mac_in_b),
        .in_c        (mac_in_c),
        .in_meta     (meta_q),
        .in_last     (1'b0),
        .out_valid   (mac_out_valid),
        .out_ready   (mac_out_ready),
        .out_result  (mac_out_result),
        .out_status  (mac_out_status),
        .out_invalid (mac_out_invalid),
        .out_meta    (),
        .out_last    ()
    );

    fp32_add_wrapper #(
        .META_W(META_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_add_eps (
        .clk         (clk),
        .rst_n       (rst_n),
        .in_valid    (add_in_valid),
        .in_ready    (add_in_ready),
        .in_a        (mean_sq_q),
        .in_b        (EPS_FP32),
        .in_meta     (meta_q),
        .in_last     (1'b0),
        .out_valid   (add_out_valid),
        .out_ready   (add_out_ready),
        .out_result  (add_out_result),
        .out_status  (add_out_status),
        .out_invalid (add_out_invalid),
        .out_meta    (),
        .out_last    ()
    );

    fp32_sqrt_wrapper #(
        .META_W(META_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_sqrt (
        .clk         (clk),
        .rst_n       (rst_n),
        .in_valid    (sqrt_in_valid),
        .in_ready    (sqrt_in_ready),
        .in_a        (add_out_result),
        .in_meta     (meta_q),
        .in_last     (1'b0),
        .out_valid   (sqrt_out_valid),
        .out_ready   (sqrt_out_ready),
        .out_result  (sqrt_out_result),
        .out_status  (sqrt_out_status),
        .out_invalid (sqrt_out_invalid),
        .out_meta    (),
        .out_last    ()
    );

    fp32_recip_wrapper #(
        .META_W(META_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_recip (
        .clk         (clk),
        .rst_n       (rst_n),
        .in_valid    (recip_in_valid),
        .in_ready    (recip_in_ready),
        .in_a        (den_q),
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

    fp16_to_fp32 #(
        .META_W(META_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_gamma_convert (
        .clk                  (clk),
        .rst_n                (rst_n),
        .in_valid             (gamma_conv_in_valid),
        .in_ready             (gamma_conv_in_ready),
        .in_data              (gamma_mem[int'(apply_index_q)]),
        .in_meta              (meta_q),
        .in_last              (1'b0),
        .out_valid            (gamma_conv_out_valid),
        .out_ready            (gamma_conv_out_ready),
        .out_data             (gamma_conv_out_data),
        .out_meta             (),
        .out_last             (),
        .out_invalid          (gamma_conv_out_invalid),
        .out_underflow_or_ftz ()
    );

    fp32_to_fp16 #(
        .META_W(META_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_quant (
        .clk                  (clk),
        .rst_n                (rst_n),
        .in_valid             (quant_in_valid),
        .in_ready             (quant_in_ready),
        .in_data              (norm_fp32_q),
        .in_meta              (meta_q),
        .in_last              (apply_last),
        .out_valid            (quant_out_valid),
        .out_ready            (quant_out_ready),
        .out_data             (quant_out_data),
        .out_invalid          (quant_out_invalid),
        .out_overflow         (quant_out_overflow),
        .out_underflow_or_ftz (quant_out_underflow_or_ftz),
        .out_inexact          (quant_out_inexact),
        .out_meta             (),
        .out_last             ()
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            input_loaded_mask_q <= '0;
            gamma_loaded_mask_q <= '0;
            input_complete_q <= 1'b0;
            gamma_complete_q <= 1'b0;
            load_error_q <= 1'b0;
            input_meta_q <= '0;
            reduce_index_q <= '0;
            apply_index_q <= '0;
            meta_q <= '0;
            status_q <= STATUS_OK;
            invalid_q <= 1'b0;
            acc_q <= FP32_ZERO;
            mean_sq_q <= FP32_ZERO;
            den_q <= FP32_ZERO;
            inv_rms_q <= FP32_ZERO;
            gamma_fp32_q <= FP32_ZERO;
            mul1_q <= FP32_ZERO;
            norm_fp32_q <= FP32_ZERO;
            done_valid_q <= 1'b0;
            done_status_q <= STATUS_OK;
            done_invalid_q <= 1'b0;
            done_meta_q <= '0;
            perf_reduce_cycles <= '0;
            perf_apply_cycles <= '0;
            perf_sfu_stall_cycles <= '0;
            perf_output_stall_cycles <= '0;
            for (int dim = 0; dim < D_MODEL; dim++) begin
                input_mem[dim] <= FP32_ZERO;
                gamma_mem[dim] <= 16'd0;
            end
        end else begin
            if (clear) begin
                state_q <= ST_IDLE;
                input_loaded_mask_q <= '0;
                gamma_loaded_mask_q <= '0;
                input_complete_q <= 1'b0;
                gamma_complete_q <= 1'b0;
                load_error_q <= 1'b0;
                done_valid_q <= 1'b0;
                done_status_q <= STATUS_OK;
                done_invalid_q <= 1'b0;
                done_meta_q <= '0;
            end else begin
                if (done_fire) begin
                    done_valid_q <= 1'b0;
                    done_status_q <= STATUS_OK;
                    done_invalid_q <= 1'b0;
                    done_meta_q <= '0;
                    state_q <= ST_IDLE;
                end

                if (gamma_fire) begin
                    if (gamma_range_legal) begin
                        gamma_mem[int'(gamma_dim)] <= gamma_data_fp16;
                        gamma_loaded_mask_q[int'(gamma_dim)] <= 1'b1;
                    end else begin
                        load_error_q <= 1'b1;
                    end
                    if (gamma_commit) begin
                        gamma_complete_q <= 1'b1;
                    end
                end

                if (input_fire) begin
                    input_meta_q <= input_meta;
                    if (input_range_legal) begin
                        input_mem[int'(input_dim)] <= input_data_fp32;
                        input_loaded_mask_q[int'(input_dim)] <= 1'b1;
                    end else begin
                        load_error_q <= 1'b1;
                    end
                    if (input_commit) begin
                        input_complete_q <= 1'b1;
                    end
                    if (input_last && input_dim != DIM_W'(D_MODEL - 1)) begin
                        load_error_q <= 1'b1;
                    end
                end

                if (!done_fire) begin
                    unique case (state_q)
                        ST_IDLE: begin
                            if (start_fire) begin
                                meta_q <= start_meta;
                                status_q <= STATUS_OK;
                                invalid_q <= 1'b0;
                                reduce_index_q <= '0;
                                apply_index_q <= '0;
                                acc_q <= FP32_ZERO;
                                mean_sq_q <= FP32_ZERO;
                                den_q <= FP32_ZERO;
                                inv_rms_q <= FP32_ZERO;
                                if (!input_complete_q || !gamma_complete_q ||
                                    input_loaded_mask_q != {D_MODEL{1'b1}} ||
                                    gamma_loaded_mask_q != {D_MODEL{1'b1}}) begin
                                    done_valid_q <= 1'b1;
                                    done_status_q <= STATUS_INCOMPLETE;
                                    done_invalid_q <= 1'b1;
                                    done_meta_q <= start_meta;
                                    state_q <= ST_DONE;
                                end else if (load_error_q) begin
                                    done_valid_q <= 1'b1;
                                    done_status_q <= STATUS_RANGE;
                                    done_invalid_q <= 1'b1;
                                    done_meta_q <= start_meta;
                                    state_q <= ST_DONE;
                                end else begin
                                    state_q <= ST_REDUCE_SEND;
                                end
                            end
                        end

                        ST_REDUCE_SEND: begin
                            if (mac_in_ready) begin
                                state_q <= ST_REDUCE_WAIT;
                            end
                        end

                        ST_REDUCE_WAIT: begin
                            if (mac_out_fire) begin
                                acc_q <= mac_out_result;
                                status_q <= status_q | mac_out_status;
                                invalid_q <= invalid_q | mac_out_invalid;
                                if (reduce_last) begin
                                    state_q <= ST_SCALE_SEND;
                                end else begin
                                    reduce_index_q <= reduce_index_q + DIM_W'(1);
                                    state_q <= ST_REDUCE_SEND;
                                end
                            end
                        end

                        ST_SCALE_SEND: begin
                            if (mac_in_ready) begin
                                state_q <= ST_SCALE_WAIT;
                            end
                        end

                        ST_SCALE_WAIT: begin
                            if (mac_out_fire) begin
                                mean_sq_q <= mac_out_result;
                                status_q <= status_q | mac_out_status;
                                invalid_q <= invalid_q | mac_out_invalid;
                                state_q <= ST_ADD_EPS_SEND;
                            end
                        end

                        ST_ADD_EPS_SEND: begin
                            if (add_in_ready) begin
                                state_q <= ST_ADD_EPS_WAIT;
                            end
                        end

                        ST_ADD_EPS_WAIT: begin
                            if (add_out_fire) begin
                                status_q <= status_q | add_out_status;
                                invalid_q <= invalid_q | add_out_invalid;
                                state_q <= ST_SQRT_SEND;
                            end
                        end

                        ST_SQRT_SEND: begin
                            if (sqrt_in_ready) begin
                                state_q <= ST_SQRT_WAIT;
                            end
                        end

                        ST_SQRT_WAIT: begin
                            if (sqrt_out_fire) begin
                                den_q <= sqrt_out_result;
                                status_q <= status_q | sqrt_out_status;
                                invalid_q <= invalid_q | sqrt_out_invalid;
                                state_q <= ST_RECIP_SEND;
                            end
                        end

                        ST_RECIP_SEND: begin
                            if (recip_in_ready) begin
                                state_q <= ST_RECIP_WAIT;
                            end
                        end

                        ST_RECIP_WAIT: begin
                            if (recip_out_fire) begin
                                inv_rms_q <= recip_out_result;
                                status_q <= status_q | recip_out_status;
                                invalid_q <= invalid_q | recip_out_invalid;
                                apply_index_q <= '0;
                                state_q <= ST_GAMMA_SEND;
                            end
                        end

                        ST_GAMMA_SEND: begin
                            if (gamma_conv_in_ready) begin
                                state_q <= ST_GAMMA_WAIT;
                            end
                        end

                        ST_GAMMA_WAIT: begin
                            if (gamma_conv_out_fire) begin
                                gamma_fp32_q <= gamma_conv_out_data;
                                invalid_q <= invalid_q | gamma_conv_out_invalid;
                                state_q <= ST_MUL1_SEND;
                            end
                        end

                        ST_MUL1_SEND: begin
                            if (mac_in_ready) begin
                                state_q <= ST_MUL1_WAIT;
                            end
                        end

                        ST_MUL1_WAIT: begin
                            if (mac_out_fire) begin
                                mul1_q <= mac_out_result;
                                status_q <= status_q | mac_out_status;
                                invalid_q <= invalid_q | mac_out_invalid;
                                state_q <= ST_MUL2_SEND;
                            end
                        end

                        ST_MUL2_SEND: begin
                            if (mac_in_ready) begin
                                state_q <= ST_MUL2_WAIT;
                            end
                        end

                        ST_MUL2_WAIT: begin
                            if (mac_out_fire) begin
                                status_q <= status_q | mac_out_status;
                                invalid_q <= invalid_q | mac_out_invalid;
                                norm_fp32_q <= mac_out_result;
                                state_q <= ST_QUANT_SEND;
                            end
                        end

                        ST_QUANT_SEND: begin
                            if (quant_in_ready) begin
                                state_q <= ST_QUANT_WAIT;
                            end
                        end

                        ST_QUANT_WAIT: begin
                            if (output_valid && !output_ready) begin
                                perf_output_stall_cycles <= perf_output_stall_cycles + COUNTER_W'(1);
                            end
                            if (quant_out_fire) begin
                                invalid_q <= invalid_q | quant_out_invalid;
                                if (apply_last) begin
                                    done_valid_q <= 1'b1;
                                    done_status_q <= status_q;
                                    done_invalid_q <= invalid_q | quant_out_invalid;
                                    done_meta_q <= meta_q;
                                    state_q <= ST_DONE;
                                end else begin
                                    apply_index_q <= apply_index_q + DIM_W'(1);
                                    state_q <= ST_GAMMA_SEND;
                                end
                            end
                        end

                        ST_DONE: begin
                            if (!done_valid_q) begin
                                state_q <= ST_IDLE;
                            end
                        end

                        default: state_q <= ST_IDLE;
                    endcase
                end

                if (state_q == ST_REDUCE_SEND || state_q == ST_REDUCE_WAIT ||
                    state_q == ST_SCALE_SEND || state_q == ST_SCALE_WAIT) begin
                    perf_reduce_cycles <= perf_reduce_cycles + COUNTER_W'(1);
                end
                if (state_q == ST_GAMMA_SEND || state_q == ST_GAMMA_WAIT ||
                    state_q == ST_MUL1_SEND || state_q == ST_MUL1_WAIT ||
                    state_q == ST_MUL2_SEND || state_q == ST_MUL2_WAIT ||
                    state_q == ST_QUANT_SEND || state_q == ST_QUANT_WAIT) begin
                    perf_apply_cycles <= perf_apply_cycles + COUNTER_W'(1);
                end
                if ((state_q == ST_SQRT_SEND && !sqrt_in_ready) ||
                    (state_q == ST_RECIP_SEND && !recip_in_ready)) begin
                    perf_sfu_stall_cycles <= perf_sfu_stall_cycles + COUNTER_W'(1);
                end
            end
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n) begin
            assert (D_MODEL == 8 || D_MODEL == 16 || D_MODEL == 32 ||
                    D_MODEL == 64 || D_MODEL == 128)
                else $error("rmsnorm_engine supported_power_of_two_d_model failed");
            if (output_valid) begin
                assert (!$isunknown({output_dim, output_data_fp16, output_status,
                                     output_invalid, output_meta, output_last}))
                    else $error("rmsnorm_engine no_unknown_output_when_valid failed");
            end
            if ($past(rst_n) && $past(output_valid && !output_ready)) begin
                assert (output_valid)
                    else $error("rmsnorm_engine output valid dropped under backpressure");
                assert ($stable({output_dim, output_data_fp16, output_status,
                                 output_invalid, output_meta, output_last}))
                    else $error("rmsnorm_engine output stable until ready failed");
            end
        end
    end
`endif
endmodule

`default_nettype wire
