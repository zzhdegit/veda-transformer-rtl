`default_nettype none

module residual_add_engine #(
    parameter int D_MODEL = 16,
    parameter int META_W = 16,
    parameter int COUNTER_W = 64,
    parameter bit ASSERT_ON_INVALID = 1'b1,
    localparam int DIM_W = (D_MODEL <= 1) ? 1 : $clog2(D_MODEL)
) (
    input  logic                         clk,
    input  logic                         rst_n,

    input  logic                         clear,

    input  logic                         input_valid,
    output logic                         input_ready,
    input  logic [DIM_W-1:0]             input_dim,
    input  logic [31:0]                  input_lhs_fp32,
    input  logic [31:0]                  input_rhs_fp32,
    input  logic                         input_last,
    input  logic [META_W-1:0]            input_meta,
    input  logic                         input_commit,

    input  logic                         start_valid,
    output logic                         start_ready,
    input  logic [META_W-1:0]            start_meta,

    output logic                         output_valid,
    input  logic                         output_ready,
    output logic [DIM_W-1:0]             output_dim,
    output logic [31:0]                  output_data_fp32,
    output logic [7:0]                   output_status,
    output logic                         output_invalid,
    output logic [META_W-1:0]            output_meta,
    output logic                         output_last,

    output logic                         done_valid,
    input  logic                         done_ready,
    output logic [7:0]                   done_status,
    output logic                         done_invalid,
    output logic [META_W-1:0]            done_meta,

    output logic [COUNTER_W-1:0]         perf_add_cycles,
    output logic [COUNTER_W-1:0]         perf_output_stall_cycles
);
    localparam logic [7:0] STATUS_OK = 8'h00;
    localparam logic [7:0] STATUS_INCOMPLETE = 8'hC1;
    localparam logic [7:0] STATUS_RANGE = 8'hC2;

    typedef enum logic [2:0] {
        ST_IDLE,
        ST_SEND,
        ST_WAIT,
        ST_DONE
    } state_e;

    state_e state_q;

    logic [31:0] lhs_mem [0:D_MODEL-1];
    logic [31:0] rhs_mem [0:D_MODEL-1];
    logic [D_MODEL-1:0] loaded_mask_q;
    logic complete_q;
    logic load_error_q;
    logic [DIM_W-1:0] index_q;
    logic [META_W-1:0] meta_q;
    logic [7:0] status_q;
    logic invalid_q;

    logic done_valid_q;
    logic [7:0] done_status_q;
    logic done_invalid_q;
    logic [META_W-1:0] done_meta_q;

    logic add_in_valid;
    logic add_in_ready;
    logic add_out_valid;
    logic add_out_ready;
    logic [31:0] add_out_result;
    logic [7:0] add_out_status;
    logic add_out_invalid;

    wire input_fire = input_valid && input_ready;
    wire start_fire = start_valid && start_ready;
    wire done_fire = done_valid && done_ready;
    wire add_out_fire = add_out_valid && add_out_ready;
    wire input_range_legal = int'(input_dim) < D_MODEL;
    wire output_last_dim = index_q == DIM_W'(D_MODEL - 1);

    initial begin
        if (D_MODEL <= 0 || META_W <= 0 || COUNTER_W <= 0) begin
            $fatal(1, "residual_add_engine parameters must be positive");
        end
    end

    assign input_ready = state_q == ST_IDLE;
    assign start_ready = (state_q == ST_IDLE) && !done_valid_q;
    assign add_in_valid = state_q == ST_SEND;
    assign add_out_ready = (state_q == ST_WAIT) && output_ready;

    assign output_valid = (state_q == ST_WAIT) && add_out_valid;
    assign output_dim = index_q;
    assign output_data_fp32 = add_out_result;
    assign output_status = status_q | add_out_status;
    assign output_invalid = invalid_q | add_out_invalid;
    assign output_meta = meta_q;
    assign output_last = output_last_dim;

    assign done_valid = done_valid_q;
    assign done_status = done_status_q;
    assign done_invalid = done_invalid_q;
    assign done_meta = done_meta_q;

    fp32_add_wrapper #(
        .META_W(META_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_add (
        .clk         (clk),
        .rst_n       (rst_n),
        .in_valid    (add_in_valid),
        .in_ready    (add_in_ready),
        .in_a        (lhs_mem[int'(index_q)]),
        .in_b        (rhs_mem[int'(index_q)]),
        .in_meta     (meta_q),
        .in_last     (output_last_dim),
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
            loaded_mask_q <= '0;
            complete_q <= 1'b0;
            load_error_q <= 1'b0;
            index_q <= '0;
            meta_q <= '0;
            status_q <= STATUS_OK;
            invalid_q <= 1'b0;
            done_valid_q <= 1'b0;
            done_status_q <= STATUS_OK;
            done_invalid_q <= 1'b0;
            done_meta_q <= '0;
            perf_add_cycles <= '0;
            perf_output_stall_cycles <= '0;
            for (int dim = 0; dim < D_MODEL; dim++) begin
                lhs_mem[dim] <= 32'd0;
                rhs_mem[dim] <= 32'd0;
            end
        end else begin
            if (clear) begin
                state_q <= ST_IDLE;
                loaded_mask_q <= '0;
                complete_q <= 1'b0;
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

                if (input_fire) begin
                    meta_q <= input_meta;
                    if (input_range_legal) begin
                        lhs_mem[int'(input_dim)] <= input_lhs_fp32;
                        rhs_mem[int'(input_dim)] <= input_rhs_fp32;
                        loaded_mask_q[int'(input_dim)] <= 1'b1;
                    end else begin
                        load_error_q <= 1'b1;
                    end
                    if (input_commit) begin
                        complete_q <= 1'b1;
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
                                index_q <= '0;
                                if (!complete_q || loaded_mask_q != {D_MODEL{1'b1}}) begin
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
                                    state_q <= ST_SEND;
                                end
                            end
                        end

                        ST_SEND: begin
                            if (add_in_ready) begin
                                state_q <= ST_WAIT;
                            end
                        end

                        ST_WAIT: begin
                            if (output_valid && !output_ready) begin
                                perf_output_stall_cycles <= perf_output_stall_cycles + COUNTER_W'(1);
                            end
                            if (add_out_fire) begin
                                status_q <= status_q | add_out_status;
                                invalid_q <= invalid_q | add_out_invalid;
                                if (output_last_dim) begin
                                    done_valid_q <= 1'b1;
                                    done_status_q <= status_q | add_out_status;
                                    done_invalid_q <= invalid_q | add_out_invalid;
                                    done_meta_q <= meta_q;
                                    state_q <= ST_DONE;
                                end else begin
                                    index_q <= index_q + DIM_W'(1);
                                    state_q <= ST_SEND;
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

                if (state_q == ST_SEND || state_q == ST_WAIT) begin
                    perf_add_cycles <= perf_add_cycles + COUNTER_W'(1);
                end
            end
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n) begin
            if (output_valid) begin
                assert (!$isunknown({output_dim, output_data_fp32, output_status,
                                     output_invalid, output_meta, output_last}))
                    else $error("residual_add_engine no_unknown_output_when_valid failed");
            end
            if ($past(rst_n) && $past(output_valid && !output_ready)) begin
                assert (output_valid)
                    else $error("residual_add_engine output valid dropped under backpressure");
                assert ($stable({output_dim, output_data_fp32, output_status,
                                 output_invalid, output_meta, output_last}))
                    else $error("residual_add_engine output stable until ready failed");
            end
        end
    end
`endif
endmodule

`default_nettype wire
