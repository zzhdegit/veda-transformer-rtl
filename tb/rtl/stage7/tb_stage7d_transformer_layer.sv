`timescale 1ns/1ps
`default_nettype none

module tb_stage7d_transformer_layer;
`ifndef STAGE7_N_HEAD
    localparam int N_HEAD = 1;
`else
    localparam int N_HEAD = `STAGE7_N_HEAD;
`endif
`ifndef STAGE7_D_HEAD
    localparam int D_HEAD = 8;
`else
    localparam int D_HEAD = `STAGE7_D_HEAD;
`endif
    localparam int D_MODEL = N_HEAD * D_HEAD;
    localparam int D_FFN = 4 * D_MODEL;
    localparam int PE_NUM = 8;
    localparam int MAX_SEQ_LEN = 8;
    localparam int META_W = 16;
    localparam int COUNTER_W = 64;
    localparam int MODEL_W = (D_MODEL <= 1) ? 1 : $clog2(D_MODEL);
    localparam int FFN_W = (D_FFN <= 1) ? 1 : $clog2(D_FFN);
    localparam int SEQ_LEN_W = (MAX_SEQ_LEN <= 1) ? 1 : $clog2(MAX_SEQ_LEN + 1);
    localparam int MAX_WEIGHT_LINES = 20000;
    localparam int MAX_TOKENS = 4;
    localparam int L_ST_LOAD_INPUT = 0;
    localparam int L_ST_NORM1_RUN = 2;
    localparam int L_ST_MHA_RUN = 3;
    localparam int L_ST_RES1_RUN = 6;
    localparam int L_ST_NORM2_RUN = 8;
    localparam int L_ST_FFN_RUN = 10;
    localparam int L_ST_RES2_RUN = 12;
    localparam int R_ST_REDUCE_SEND = 1;
    localparam int R_ST_REDUCE_WAIT = 2;
    localparam int R_ST_GAMMA_SEND = 11;
    localparam int R_ST_QUANT_WAIT = 18;
    localparam int F_ST_FFN1_SEND_TILE = 1;
    localparam int F_ST_FFN1_WAIT_OUTPUT = 2;
    localparam int F_ST_RELU_QUANT_SEND = 3;
    localparam int F_ST_RELU_QUANT_WAIT = 4;
    localparam int F_ST_FFN2_SEND_TILE = 5;
    localparam int F_ST_FFN2_WAIT_OUTPUT = 6;
    localparam int A_ST_SEND = 1;
    localparam int A_ST_WAIT = 2;

    logic clk;
    logic rst_n;

    logic weight_valid;
    logic weight_ready;
    logic [2:0] weight_kind;
    logic [FFN_W-1:0] weight_output_index;
    logic [FFN_W-1:0] weight_input_index;
    logic [15:0] weight_data_fp16;
    logic weight_last;
    logic weight_commit;
    logic token_valid;
    logic token_ready;
    logic [MODEL_W-1:0] token_dim;
    logic [15:0] token_hidden_fp16;
    logic token_last_dim;
    logic [META_W-1:0] token_meta;
    logic output_valid;
    logic output_ready;
    logic [MODEL_W-1:0] output_base_dim;
    logic [PE_NUM*32-1:0] output_vector_fp32;
    logic [PE_NUM-1:0] output_lane_mask;
    logic [7:0] output_status;
    logic output_invalid;
    logic [META_W-1:0] output_meta;
    logic output_last;
    logic done_valid;
    logic done_ready;
    logic [7:0] done_status;
    logic done_invalid;
    logic [META_W-1:0] done_meta;
    logic [SEQ_LEN_W-1:0] done_valid_seq_len;
    logic [SEQ_LEN_W-1:0] current_valid_seq_len;

    logic [COUNTER_W-1:0] perf_generation_steps;
    logic [COUNTER_W-1:0] perf_total_layer_cycles;
    logic [COUNTER_W-1:0] perf_input_load_cycles;
    logic [COUNTER_W-1:0] perf_norm1_reduce_cycles;
    logic [COUNTER_W-1:0] perf_norm1_apply_cycles;
    logic [COUNTER_W-1:0] perf_mha_cycles;
    logic [COUNTER_W-1:0] perf_residual1_cycles;
    logic [COUNTER_W-1:0] perf_norm2_reduce_cycles;
    logic [COUNTER_W-1:0] perf_norm2_apply_cycles;
    logic [COUNTER_W-1:0] perf_ffn1_cycles;
    logic [COUNTER_W-1:0] perf_relu_cycles;
    logic [COUNTER_W-1:0] perf_activation_quantization_cycles;
    logic [COUNTER_W-1:0] perf_ffn2_cycles;
    logic [COUNTER_W-1:0] perf_residual2_cycles;
    logic [COUNTER_W-1:0] perf_final_output_cycles;
    logic [COUNTER_W-1:0] perf_norm_stall_cycles;
    logic [COUNTER_W-1:0] perf_mha_stall_cycles;
    logic [COUNTER_W-1:0] perf_ffn_pe_stall_cycles;
    logic [COUNTER_W-1:0] perf_weight_stall_cycles;
    logic [COUNTER_W-1:0] perf_buffer_stall_cycles;
    logic [COUNTER_W-1:0] perf_output_stall_cycles;
    logic [SEQ_LEN_W-1:0] perf_peak_valid_seq_len;

    int weight_count;
    int weight_kind_vec [0:MAX_WEIGHT_LINES-1];
    int weight_row_vec [0:MAX_WEIGHT_LINES-1];
    int weight_col_vec [0:MAX_WEIGHT_LINES-1];
    logic [15:0] weight_data_vec [0:MAX_WEIGHT_LINES-1];
    logic [15:0] hidden_vec [0:MAX_TOKENS-1][0:D_MODEL-1];
    logic [31:0] expected_output [0:MAX_TOKENS-1][0:D_MODEL-1];
    logic [META_W-1:0] expected_meta [0:MAX_TOKENS-1];
    int token_count;

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    transformer_layer #(
        .N_HEAD(N_HEAD),
        .D_HEAD(D_HEAD),
        .PE_NUM(PE_NUM),
        .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .META_W(META_W),
        .COUNTER_W(COUNTER_W)
    ) u_layer (
        .clk                                  (clk),
        .rst_n                                (rst_n),
        .weight_valid                         (weight_valid),
        .weight_ready                         (weight_ready),
        .weight_kind                          (weight_kind),
        .weight_output_index                  (weight_output_index),
        .weight_input_index                   (weight_input_index),
        .weight_data_fp16                     (weight_data_fp16),
        .weight_last                          (weight_last),
        .weight_commit                        (weight_commit),
        .token_valid                          (token_valid),
        .token_ready                          (token_ready),
        .token_dim                            (token_dim),
        .token_hidden_fp16                    (token_hidden_fp16),
        .token_last_dim                       (token_last_dim),
        .token_meta                           (token_meta),
        .output_valid                         (output_valid),
        .output_ready                         (output_ready),
        .output_base_dim                      (output_base_dim),
        .output_vector_fp32                   (output_vector_fp32),
        .output_lane_mask                     (output_lane_mask),
        .output_status                        (output_status),
        .output_invalid                       (output_invalid),
        .output_meta                          (output_meta),
        .output_last                          (output_last),
        .done_valid                           (done_valid),
        .done_ready                           (done_ready),
        .done_status                          (done_status),
        .done_invalid                         (done_invalid),
        .done_meta                            (done_meta),
        .done_valid_seq_len                   (done_valid_seq_len),
        .current_valid_seq_len                (current_valid_seq_len),
        .perf_generation_steps                (perf_generation_steps),
        .perf_total_layer_cycles              (perf_total_layer_cycles),
        .perf_input_load_cycles               (perf_input_load_cycles),
        .perf_norm1_reduce_cycles             (perf_norm1_reduce_cycles),
        .perf_norm1_apply_cycles              (perf_norm1_apply_cycles),
        .perf_mha_cycles                      (perf_mha_cycles),
        .perf_residual1_cycles                (perf_residual1_cycles),
        .perf_norm2_reduce_cycles             (perf_norm2_reduce_cycles),
        .perf_norm2_apply_cycles              (perf_norm2_apply_cycles),
        .perf_ffn1_cycles                     (perf_ffn1_cycles),
        .perf_relu_cycles                     (perf_relu_cycles),
        .perf_activation_quantization_cycles  (perf_activation_quantization_cycles),
        .perf_ffn2_cycles                     (perf_ffn2_cycles),
        .perf_residual2_cycles                (perf_residual2_cycles),
        .perf_final_output_cycles             (perf_final_output_cycles),
        .perf_norm_stall_cycles               (perf_norm_stall_cycles),
        .perf_mha_stall_cycles                (perf_mha_stall_cycles),
        .perf_ffn_pe_stall_cycles             (perf_ffn_pe_stall_cycles),
        .perf_weight_stall_cycles             (perf_weight_stall_cycles),
        .perf_buffer_stall_cycles             (perf_buffer_stall_cycles),
        .perf_output_stall_cycles             (perf_output_stall_cycles),
        .perf_peak_valid_seq_len              (perf_peak_valid_seq_len)
    );

    task automatic tb_fail(input string message);
        begin
            $display("STAGE7D_TB_FAIL: %s", message);
            $fatal(1);
        end
    endtask

    task automatic load_vectors;
        string path;
        int fd;
        int code;
        string tag;
        int n_head_value;
        int d_head_value;
        int d_model_value;
        int d_ffn_value;
        int kind;
        int row;
        int col;
        int dim;
        int token_idx;
        int current_token;
        logic [15:0] h;
        logic [31:0] f;
        logic [META_W-1:0] meta_value;
        begin
            if (!$value$plusargs("STAGE7D_VECTOR_FILE=%s", path)) begin
                tb_fail("missing +STAGE7D_VECTOR_FILE");
            end
            fd = $fopen(path, "r");
            if (fd == 0) begin
                tb_fail("could not open vector file");
            end
            weight_count = 0;
            token_count = 0;
            current_token = 0;
            for (int tok = 0; tok < MAX_TOKENS; tok++) begin
                expected_meta[tok] = META_W'(16'h7D01 + tok);
            end
            while (!$feof(fd)) begin
                code = $fscanf(fd, "%s", tag);
                if (code == 1) begin
                    if (tag == "C") begin
                        code = $fscanf(fd, "%d %d %d %d\n", n_head_value, d_head_value, d_model_value, d_ffn_value);
                        if (n_head_value != N_HEAD || d_head_value != D_HEAD ||
                            d_model_value != D_MODEL || d_ffn_value != D_FFN) begin
                            tb_fail("configuration mismatch");
                        end
                    end else if (tag == "W") begin
                        code = $fscanf(fd, "%d %h %h %h\n", kind, row, col, h);
                        if (weight_count >= MAX_WEIGHT_LINES) tb_fail("too many weight lines");
                        weight_kind_vec[weight_count] = kind;
                        weight_row_vec[weight_count] = row;
                        weight_col_vec[weight_count] = col;
                        weight_data_vec[weight_count] = h;
                        weight_count++;
                    end else if (tag == "T") begin
                        code = $fscanf(fd, "%d %h\n", token_idx, meta_value);
                        if (token_idx < 0 || token_idx >= MAX_TOKENS) tb_fail("token index out of range");
                        current_token = token_idx;
                        expected_meta[current_token] = meta_value;
                        if (token_count < current_token + 1) token_count = current_token + 1;
                    end else if (tag == "H") begin
                        code = $fscanf(fd, "%h %h\n", dim, h);
                        hidden_vec[current_token][dim] = h;
                        if (token_count == 0) token_count = 1;
                    end else if (tag == "O") begin
                        code = $fscanf(fd, "%h %h\n", dim, f);
                        expected_output[current_token][dim] = f;
                        if (token_count == 0) token_count = 1;
                    end else begin
                        tb_fail("unknown vector tag");
                    end
                end
            end
            $fclose(fd);
        end
    endtask

    task automatic apply_reset;
        begin
            rst_n = 1'b0;
            weight_valid = 1'b0;
            weight_kind = '0;
            weight_output_index = '0;
            weight_input_index = '0;
            weight_data_fp16 = '0;
            weight_last = 1'b0;
            weight_commit = 1'b0;
            token_valid = 1'b0;
            token_dim = '0;
            token_hidden_fp16 = '0;
            token_last_dim = 1'b0;
            token_meta = '0;
            output_ready = 1'b0;
            done_ready = 1'b0;
            repeat (8) @(posedge clk);
            rst_n = 1'b1;
            repeat (4) @(posedge clk);
        end
    endtask

    task automatic pulse_reset_and_check(input string scenario);
        begin
            @(negedge clk);
            rst_n = 1'b0;
            weight_valid = 1'b0;
            weight_kind = '0;
            weight_output_index = '0;
            weight_input_index = '0;
            weight_data_fp16 = '0;
            weight_last = 1'b0;
            weight_commit = 1'b0;
            token_valid = 1'b0;
            token_dim = '0;
            token_hidden_fp16 = '0;
            token_last_dim = 1'b0;
            token_meta = '0;
            output_ready = 1'b0;
            done_ready = 1'b0;
            repeat (4) @(posedge clk);
            #1;
            if (output_valid || done_valid) tb_fail({scenario, ": valid high during reset"});
            if ($isunknown({output_valid, done_valid, output_status, output_invalid,
                            done_status, done_invalid, output_meta, done_meta,
                            current_valid_seq_len})) begin
                tb_fail({scenario, ": X on reset-visible status"});
            end
            if (output_meta !== '0 || done_meta !== '0) tb_fail({scenario, ": metadata leaked after reset"});
            rst_n = 1'b1;
            repeat (4) @(posedge clk);
            #1;
            if (output_valid || done_valid) tb_fail({scenario, ": stale valid after reset release"});
            if (current_valid_seq_len !== '0) tb_fail({scenario, ": valid_seq_len not reset"});
        end
    endtask

    task automatic wait_layer_state(input int target, input string scenario);
        bit seen;
        begin
            seen = 1'b0;
            for (int cycle = 0; cycle < 200000; cycle++) begin
                @(negedge clk);
                #1;
                if (int'(u_layer.state_q) == target) begin
                    seen = 1'b1;
                    break;
                end
            end
            if (!seen) tb_fail({scenario, ": target layer state not reached"});
        end
    endtask

    task automatic wait_norm_state(input bit norm2, input int lo, input int hi, input string scenario);
        bit seen;
        int state_value;
        begin
            seen = 1'b0;
            for (int cycle = 0; cycle < 200000; cycle++) begin
                @(negedge clk);
                #1;
                state_value = norm2 ? int'(u_layer.u_norm2.state_q) : int'(u_layer.u_norm1.state_q);
                if (state_value >= lo && state_value <= hi) begin
                    seen = 1'b1;
                    break;
                end
            end
            if (!seen) tb_fail({scenario, ": target RMSNorm state not reached"});
        end
    endtask

    task automatic wait_ffn_state(input int lo, input int hi, input string scenario);
        bit seen;
        int state_value;
        begin
            seen = 1'b0;
            for (int cycle = 0; cycle < 300000; cycle++) begin
                @(negedge clk);
                #1;
                state_value = int'(u_layer.u_ffn.state_q);
                if (state_value >= lo && state_value <= hi) begin
                    seen = 1'b1;
                    break;
                end
            end
            if (!seen) tb_fail({scenario, ": target FFN state not reached"});
        end
    endtask

    task automatic wait_residual_state(input bit residual2, input int lo, input int hi, input string scenario);
        bit seen;
        int state_value;
        begin
            seen = 1'b0;
            for (int cycle = 0; cycle < 200000; cycle++) begin
                @(negedge clk);
                #1;
                state_value = residual2 ? int'(u_layer.u_residual2.state_q) : int'(u_layer.u_residual1.state_q);
                if (state_value >= lo && state_value <= hi) begin
                    seen = 1'b1;
                    break;
                end
            end
            if (!seen) tb_fail({scenario, ": target residual state not reached"});
        end
    endtask

    task automatic load_partial_token(input int token_idx, input int dims);
        begin
            for (int dim = 0; dim < dims; dim++) begin
                @(negedge clk);
                token_valid = 1'b1;
                token_dim = MODEL_W'(dim);
                token_hidden_fp16 = hidden_vec[token_idx][dim];
                token_last_dim = 1'b0;
                token_meta = expected_meta[token_idx];
                #1;
                if (!token_ready) tb_fail("partial token load not ready");
                @(posedge clk);
            end
            @(negedge clk);
            token_valid = 1'b0;
        end
    endtask

    task automatic start_token_for_reset(input int token_idx);
        begin
            output_ready = 1'b1;
            done_ready = 1'b1;
            load_token(token_idx);
            output_ready = 1'b1;
            done_ready = 1'b1;
        end
    endtask

    task automatic recover_clean_token(input string scenario);
        begin
            load_weights();
            run_layer(0);
            if (perf_generation_steps !== COUNTER_W'(1)) tb_fail({scenario, ": duplicate or missing recovery commit"});
            $display("STAGE7D_RESET_SCENARIO_PASS %s", scenario);
        end
    endtask

    task automatic reset_during_layer_state(input string scenario, input int target_state);
        begin
            apply_reset();
            load_weights();
            start_token_for_reset(0);
            wait_layer_state(target_state, scenario);
            pulse_reset_and_check(scenario);
            recover_clean_token(scenario);
        end
    endtask

    task automatic reset_during_norm(input string scenario, input bit norm2, input int lo, input int hi);
        begin
            apply_reset();
            load_weights();
            start_token_for_reset(0);
            wait_norm_state(norm2, lo, hi, scenario);
            pulse_reset_and_check(scenario);
            recover_clean_token(scenario);
        end
    endtask

    task automatic reset_during_ffn(input string scenario, input int lo, input int hi);
        begin
            apply_reset();
            load_weights();
            start_token_for_reset(0);
            wait_ffn_state(lo, hi, scenario);
            pulse_reset_and_check(scenario);
            recover_clean_token(scenario);
        end
    endtask

    task automatic reset_during_residual(input string scenario, input bit residual2);
        begin
            apply_reset();
            load_weights();
            start_token_for_reset(0);
            wait_residual_state(residual2, A_ST_SEND, A_ST_WAIT, scenario);
            pulse_reset_and_check(scenario);
            recover_clean_token(scenario);
        end
    endtask

    task automatic load_weights;
        logic group_last;
        logic stage6_kind;
        begin
            for (int idx = 0; idx < weight_count; idx++) begin
                group_last = (idx == weight_count - 1) || (weight_kind_vec[idx + 1] != weight_kind_vec[idx]);
                stage6_kind = weight_kind_vec[idx] <= 3;
                @(negedge clk);
                weight_valid = 1'b1;
                weight_kind = 3'(weight_kind_vec[idx]);
                weight_output_index = FFN_W'(weight_row_vec[idx]);
                weight_input_index = FFN_W'(weight_col_vec[idx]);
                weight_data_fp16 = weight_data_vec[idx];
                weight_last = group_last;
                weight_commit = group_last && !stage6_kind;
                #1;
                if (!weight_ready) tb_fail("weight load not ready");
                @(posedge clk);
                @(negedge clk);
                weight_valid = 1'b0;
                weight_last = 1'b0;
                weight_commit = 1'b0;
                if (group_last && stage6_kind) begin
                    weight_kind = 3'(weight_kind_vec[idx]);
                    @(posedge clk);
                    @(negedge clk);
                    weight_commit = 1'b1;
                    @(posedge clk);
                    @(negedge clk);
                    weight_commit = 1'b0;
                end
            end
        end
    endtask

    task automatic load_token(input int token_idx);
        begin
            for (int dim = 0; dim < D_MODEL; dim++) begin
                @(negedge clk);
                token_valid = 1'b1;
                token_dim = MODEL_W'(dim);
                token_hidden_fp16 = hidden_vec[token_idx][dim];
                token_last_dim = dim == D_MODEL - 1;
                token_meta = expected_meta[token_idx];
                #1;
                if (!token_ready) tb_fail("token load not ready");
                @(posedge clk);
            end
            @(negedge clk);
            token_valid = 1'b0;
            token_last_dim = 1'b0;
        end
    endtask

    task automatic run_layer(input int token_idx);
        int received;
        int cycle;
        logic pre_out_fire;
        logic pre_done_fire;
        int idx;
        logic [31:0] lane_value;
        bit done_seen;
        bit done_stall_started;
        int done_stall_left;
        logic [7:0] held_done_status;
        logic held_done_invalid;
        logic [META_W-1:0] held_done_meta;
        logic [SEQ_LEN_W-1:0] held_done_seq_len;
        bit held_done_active;
        begin
            load_token(token_idx);
            wait_layer_state(L_ST_NORM1_RUN, "active input/weight backpressure");
            @(negedge clk);
            token_valid = 1'b1;
            token_dim = '0;
            token_hidden_fp16 = hidden_vec[token_idx][0];
            token_last_dim = 1'b0;
            token_meta = expected_meta[token_idx];
            weight_valid = 1'b1;
            weight_kind = 3'd0;
            weight_output_index = '0;
            weight_input_index = '0;
            weight_data_fp16 = 16'd0;
            #1;
            if (token_ready || weight_ready) tb_fail("active transaction accepted input or weight");
            @(posedge clk);
            @(negedge clk);
            token_valid = 1'b0;
            weight_valid = 1'b0;
            received = 0;
            cycle = 0;
            done_seen = 1'b0;
            done_stall_started = 1'b0;
            done_stall_left = 0;
            held_done_active = 1'b0;
            while (received < D_MODEL || !done_seen) begin
                @(negedge clk);
                output_ready = (cycle % 7) != 3;
                if (done_valid && !done_stall_started) begin
                    done_ready = 1'b0;
                    done_stall_started = 1'b1;
                    done_stall_left = 2;
                    held_done_active = 1'b1;
                    held_done_status = done_status;
                    held_done_invalid = done_invalid;
                    held_done_meta = done_meta;
                    held_done_seq_len = done_valid_seq_len;
                end else if (done_stall_left > 0) begin
                    done_ready = 1'b0;
                    done_stall_left--;
                end else begin
                    done_ready = 1'b1;
                end
                #1;
                pre_out_fire = output_valid && output_ready;
                pre_done_fire = done_valid && done_ready;
                if (done_valid && !done_ready && held_done_active) begin
                    if ({done_status, done_invalid, done_meta, done_valid_seq_len} !==
                        {held_done_status, held_done_invalid, held_done_meta, held_done_seq_len}) begin
                        tb_fail("done changed under backpressure");
                    end
                end
                if (pre_out_fire) begin
                    for (int lane = 0; lane < PE_NUM; lane++) begin
                        if (output_lane_mask[lane]) begin
                            idx = int'(output_base_dim) + lane;
                            lane_value = output_vector_fp32[lane*32 +: 32];
                            if (idx >= D_MODEL) tb_fail("output lane index out of range");
                            if (lane_value !== expected_output[token_idx][idx]) begin
                                $display("CHECK_FAIL layer dim=%0d got=%08h expected=%08h",
                                         idx, lane_value, expected_output[token_idx][idx]);
                                $fatal(1);
                            end
                            received++;
                        end
                    end
                    if (output_invalid) tb_fail("output invalid");
                    if (output_meta !== expected_meta[token_idx]) tb_fail("output metadata mismatch");
                    if (output_last !== (received == D_MODEL)) tb_fail("output last mismatch");
                end
                if (pre_done_fire) begin
                    if (received != D_MODEL) tb_fail("done before all outputs");
                    if (done_invalid) tb_fail("done invalid");
                    if (done_meta !== expected_meta[token_idx]) tb_fail("done metadata mismatch");
                    if (done_valid_seq_len !== SEQ_LEN_W'(token_idx + 1)) tb_fail("done seq len mismatch");
                    done_seen = 1'b1;
                end
                @(posedge clk);
                cycle++;
                if (cycle > 300000) tb_fail("layer timeout");
            end
            @(negedge clk);
            output_ready = 1'b0;
            done_ready = 1'b0;
        end
    endtask

    task automatic reset_during_final_output_stall;
        string scenario;
        bit seen;
        begin
            scenario = "final_output_stall";
            apply_reset();
            load_weights();
            start_token_for_reset(0);
            output_ready = 1'b0;
            done_ready = 1'b1;
            seen = 1'b0;
            for (int cycle = 0; cycle < 300000; cycle++) begin
                @(negedge clk);
                #1;
                if (output_valid) begin
                    seen = 1'b1;
                    break;
                end
            end
            if (!seen) tb_fail({scenario, ": output_valid not reached"});
            pulse_reset_and_check(scenario);
            recover_clean_token(scenario);
        end
    endtask

    task automatic reset_during_done_stall;
        string scenario;
        bit seen;
        int received;
        int idx;
        logic [31:0] lane_value;
        begin
            scenario = "layer_done_stall";
            apply_reset();
            load_weights();
            load_token(0);
            output_ready = 1'b1;
            done_ready = 1'b0;
            received = 0;
            seen = 1'b0;
            for (int cycle = 0; cycle < 300000; cycle++) begin
                @(negedge clk);
                #1;
                if (output_valid && output_ready) begin
                    for (int lane = 0; lane < PE_NUM; lane++) begin
                        if (output_lane_mask[lane]) begin
                            idx = int'(output_base_dim) + lane;
                            lane_value = output_vector_fp32[lane*32 +: 32];
                            if (idx >= D_MODEL) tb_fail("done stall output lane index out of range");
                            if (lane_value !== expected_output[0][idx]) tb_fail("done stall output mismatch");
                            received++;
                        end
                    end
                end
                if (done_valid) begin
                    if (received != D_MODEL) tb_fail("done stall reached before all outputs");
                    seen = 1'b1;
                    break;
                end
            end
            if (!seen) tb_fail({scenario, ": done_valid not reached"});
            pulse_reset_and_check(scenario);
            recover_clean_token(scenario);
        end
    endtask

    task automatic run_reset_audit;
        begin
            if (token_count < 1) tb_fail("reset audit needs at least one token vector");
            apply_reset();
            load_weights();
            load_partial_token(0, (D_MODEL > 2) ? 2 : 1);
            pulse_reset_and_check("input_load");
            recover_clean_token("input_load");

            reset_during_norm("rmsnorm1_reduction", 1'b0, R_ST_REDUCE_SEND, R_ST_REDUCE_WAIT);
            reset_during_norm("rmsnorm1_apply", 1'b0, R_ST_GAMMA_SEND, R_ST_QUANT_WAIT);
            reset_during_layer_state("mha", L_ST_MHA_RUN);
            reset_during_residual("residual1", 1'b0);
            reset_during_norm("rmsnorm2_reduction", 1'b1, R_ST_REDUCE_SEND, R_ST_REDUCE_WAIT);
            reset_during_norm("rmsnorm2_apply", 1'b1, R_ST_GAMMA_SEND, R_ST_QUANT_WAIT);
            reset_during_ffn("ffn1", F_ST_FFN1_SEND_TILE, F_ST_FFN1_WAIT_OUTPUT);
            reset_during_ffn("relu", F_ST_RELU_QUANT_SEND, F_ST_RELU_QUANT_SEND);
            reset_during_ffn("activation_quantization", F_ST_RELU_QUANT_WAIT, F_ST_RELU_QUANT_WAIT);
            reset_during_ffn("ffn2", F_ST_FFN2_SEND_TILE, F_ST_FFN2_WAIT_OUTPUT);
            reset_during_residual("residual2", 1'b1);
            reset_during_final_output_stall();
            reset_during_done_stall();
            $display("STAGE7D_RESET_AUDIT_PASS scenarios=14");
        end
    endtask

    initial begin
        load_vectors();
        if ($test$plusargs("STAGE7D_RESET_AUDIT")) begin
            run_reset_audit();
        end else begin
            apply_reset();
            load_weights();
            for (int tok = 0; tok < token_count; tok++) begin
                run_layer(tok);
            end
        end
        $display("STAGE7D_TRANSFORMER_LAYER_PASS n_head=%0d d_head=%0d d_model=%0d tokens=%0d",
                 N_HEAD, D_HEAD, D_MODEL, token_count);
        $finish;
    end
endmodule

`default_nettype wire
