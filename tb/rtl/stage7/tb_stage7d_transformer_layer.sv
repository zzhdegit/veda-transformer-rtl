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
        begin
            load_token(token_idx);
            received = 0;
            cycle = 0;
            while (received < D_MODEL || !done_valid) begin
                @(negedge clk);
                output_ready = (cycle % 7) != 3;
                done_ready = 1'b1;
                #1;
                pre_out_fire = output_valid && output_ready;
                pre_done_fire = done_valid && done_ready;
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

    initial begin
        load_vectors();
        apply_reset();
        load_weights();
        for (int tok = 0; tok < token_count; tok++) begin
            run_layer(tok);
        end
        $display("STAGE7D_TRANSFORMER_LAYER_PASS n_head=%0d d_head=%0d d_model=%0d tokens=%0d",
                 N_HEAD, D_HEAD, D_MODEL, token_count);
        $finish;
    end
endmodule

`default_nettype wire
